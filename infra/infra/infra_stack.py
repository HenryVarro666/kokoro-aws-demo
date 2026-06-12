from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_ecs_patterns as patterns,
    aws_elasticloadbalancingv2 as elbv2,
    aws_sns as sns,
    aws_sns_subscriptions as subs,
)
from constructs import Construct


class InfraStack(Stack):
    def __init__(self, scope: Construct, construct_id: str,
                 avatar=None, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # nat_gateways=0: NAT costs ~$32/mo and is the classic beginner bill trap.
        # Fargate tasks sit in public subnets with public IPs to pull images directly.
        vpc = ec2.Vpc(
            self, "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                )
            ],
        )

        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)

        # CDK builds and pushes the Docker image to ECR automatically.
        image = ecs.ContainerImage.from_asset(
            "..", platform=ecr_assets.Platform.LINUX_ARM64
        )

        # Audio-quality fix: on Graviton, multi-threaded oneDNN/ACL kernels
        # (enabled once the task has >=2 vCPU) audibly degrade the vocoder
        # output (-4 dB level, spectral artifacts). Pin fp32 math and a single
        # torch thread; verified by A/B spectral comparison against local runs.
        env = {
            "DNNL_DEFAULT_FPMATH_MODE": "F32",
            # Bisection: F32 held, threads released. If audio stays clean the
            # culprit was fast-math alone and we keep multi-thread speed.
            "OMP_NUM_THREADS": "2",
        }
        if avatar is not None:
            env.update({
                "AVATAR_BUCKET": avatar.bucket.bucket_name,
                "AVATAR_QUEUE_URL": avatar.queue.queue_url,
                "AVATAR_TABLE": avatar.table.table_name,
            })

        service = patterns.ApplicationLoadBalancedFargateService(
            self, "TtsService",
            cluster=cluster,
            cpu=2048,                 # 2 vCPU (RTF experiment: 1 vCPU measured RTF 2.4)
            memory_limit_mib=4096,    # 4 GB (Fargate: 2 vCPU requires >= 4 GB)
            desired_count=1,
            # Graviton ARM64: ~20% cheaper than x86 and matches Apple Silicon builds.
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
            task_image_options=patterns.ApplicationLoadBalancedTaskImageOptions(
                image=image, container_port=8000, environment=env
            ),
            public_load_balancer=True,
            assign_public_ip=True,
            task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            # Model loads at container startup; give the ALB health check a buffer.
            health_check_grace_period=Duration.seconds(120),
        )
        service.target_group.configure_health_check(
            path="/health", healthy_threshold_count=2, interval=Duration.seconds(30)
        )

        if avatar is not None:
            task_role = service.task_definition.task_role
            avatar.bucket.grant_read_write(task_role)
            avatar.queue.grant_send_messages(task_role)
            avatar.table.grant_read_write_data(task_role)

        # Observability: alarms as code, email via SNS (confirm the subscription
        # email once after first deploy).
        alarm_topic = sns.Topic(self, "AlarmTopic")
        alarm_topic.add_subscription(
            subs.EmailSubscription("ccao2679@terpmail.umd.edu"))

        cloudwatch.Alarm(
            self, "Alb5xx",
            metric=service.load_balancer.metrics.http_code_target(
                elbv2.HttpCodeTarget.TARGET_5XX_COUNT,
                period=Duration.minutes(5), statistic="Sum"),
            threshold=1, evaluation_periods=1,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="TTS service returned 5xx",
        ).add_alarm_action(cw_actions.SnsAction(alarm_topic))

        cloudwatch.Alarm(
            self, "ServiceCpuHigh",
            metric=service.service.metric_cpu_utilization(period=Duration.minutes(5)),
            threshold=85, evaluation_periods=2,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="TTS service CPU > 85% for 10 minutes",
        ).add_alarm_action(cw_actions.SnsAction(alarm_topic))

        CfnOutput(self, "DemoUrl",
                  value=f"http://{service.load_balancer.load_balancer_dns_name}")
