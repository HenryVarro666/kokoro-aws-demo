from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_ecs_patterns as patterns,
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

        # Phase 3 (optional): point the API at the avatar pipeline resources.
        env = {}
        if avatar is not None:
            env = {
                "AVATAR_BUCKET": avatar.bucket.bucket_name,
                "AVATAR_QUEUE_URL": avatar.queue.queue_url,
                "AVATAR_TABLE": avatar.table.table_name,
            }

        service = patterns.ApplicationLoadBalancedFargateService(
            self, "TtsService",
            cluster=cluster,
            cpu=1024,                 # 1 vCPU
            memory_limit_mib=2048,    # 2 GB
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

        CfnOutput(self, "DemoUrl",
                  value=f"http://{service.load_balancer.load_balancer_dns_name}")
