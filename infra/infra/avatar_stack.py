from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_autoscaling as autoscaling,
    aws_cloudwatch as cloudwatch,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
    aws_sqs as sqs,
)
from constructs import Construct


class AvatarStack(Stack):
    """Phase 3: async GPU avatar pipeline.

    S3 (audio in / mp4 out) + SQS (jobs) + DynamoDB (status)
    + ASG of one g4dn.xlarge Spot worker that scales 0 <-> 1 on queue depth.

    Deploy with:
      cdk deploy --context avatar=true --context avatar_ami=ami-XXXX AvatarStack
    (avatar_ami = the golden AMI you bake on Day 2 of guide 03; until then the
    stack still synths using a placeholder image so you can review with `cdk diff`.)
    """

    def __init__(self, scope: Construct, construct_id: str,
                 ami_id: str | None = None, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.bucket = s3.Bucket(
            self, "Artifacts",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(7))],
        )

        dlq = sqs.Queue(self, "JobsDLQ", retention_period=Duration.days(7))
        self.queue = sqs.Queue(
            self, "Jobs",
            visibility_timeout=Duration.minutes(15),  # > max job duration
            dead_letter_queue=sqs.DeadLetterQueue(max_receive_count=3, queue=dlq),
        )

        self.table = dynamodb.Table(
            self, "JobsTable",
            partition_key=dynamodb.Attribute(
                name="job_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        vpc = ec2.Vpc(
            self, "WorkerVpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24)
            ],
        )

        role = iam.Role(self, "WorkerRole",
                        assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"))
        self.bucket.grant_read_write(role)
        self.queue.grant_consume_messages(role)
        self.table.grant_read_write_data(role)
        # worker scales itself to 0 when idle
        role.add_to_policy(iam.PolicyStatement(
            actions=["autoscaling:SetDesiredCapacity"], resources=["*"]))
        # SSM Session Manager: shell into the worker without opening SSH
        role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name(
            "AmazonSSMManagedInstanceCore"))

        if ami_id:
            machine_image = ec2.MachineImage.generic_linux({self.region: ami_id})
        else:
            # placeholder so `cdk synth/diff` works before the golden AMI exists
            machine_image = ec2.MachineImage.latest_amazon_linux2023()

        # The golden AMI ships a systemd unit (avatar-worker.service) that reads
        # /etc/avatar-worker.env; user-data just writes the env and starts it.
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "cat > /etc/avatar-worker.env <<EOF\n"
            f"AVATAR_QUEUE_URL={self.queue.queue_url}\n"
            f"AVATAR_BUCKET={self.bucket.bucket_name}\n"
            f"AVATAR_TABLE={self.table.table_name}\n"
            f"AWS_DEFAULT_REGION={self.region}\n"
            "ASG_NAME=avatar-worker\n"
            "EOF",
            "systemctl restart avatar-worker || true",
        )

        asg = autoscaling.AutoScalingGroup(
            self, "WorkerAsg",
            auto_scaling_group_name="avatar-worker",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType("g4dn.xlarge"),  # T4 16GB
            machine_image=machine_image,
            role=role,
            min_capacity=0,
            max_capacity=1,
            spot_price="0.45",  # cap; on-demand is $0.526/h, spot usually $0.15-0.25
            associate_public_ip_address=True,
            user_data=user_data,
        )

        # Scale on total queue depth (visible + in-flight). Including in-flight
        # messages prevents scale-in from killing a worker mid-job.
        visible = self.queue.metric_approximate_number_of_messages_visible(
            period=Duration.minutes(1), statistic="Maximum")
        in_flight = self.queue.metric_approximate_number_of_messages_not_visible(
            period=Duration.minutes(1), statistic="Maximum")
        total = cloudwatch.MathExpression(
            expression="v + n", using_metrics={"v": visible, "n": in_flight},
            period=Duration.minutes(1))
        asg.scale_on_metric(
            "QueueDepth",
            metric=total,
            adjustment_type=autoscaling.AdjustmentType.EXACT_CAPACITY,
            scaling_steps=[
                autoscaling.ScalingInterval(upper=1, change=0),  # queue empty -> 0
                autoscaling.ScalingInterval(lower=1, change=1),  # work waiting -> 1
            ],
        )

        CfnOutput(self, "BucketName", value=self.bucket.bucket_name)
        CfnOutput(self, "QueueUrl", value=self.queue.queue_url)
        CfnOutput(self, "TableName", value=self.table.table_name)
