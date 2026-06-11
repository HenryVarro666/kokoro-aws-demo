from aws_cdk import CfnOutput, Stack, aws_iam as iam
from constructs import Construct

GITHUB_REPO = "HenryVarro666/kokoro-aws-demo"


class OidcStack(Stack):
    """One-time stack: lets GitHub Actions assume an AWS role via OIDC (no stored keys).

    Deploy once from your laptop: `cdk deploy OidcStack`, then put the output
    role ARN into .github/workflows/deploy.yml.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        provider = iam.OpenIdConnectProvider(
            self, "GithubOidc",
            url="https://token.actions.githubusercontent.com",
            client_ids=["sts.amazonaws.com"],
        )

        role = iam.Role(
            self, "GithubDeployRole",
            role_name="github-deploy-kokoro",
            assumed_by=iam.WebIdentityPrincipal(
                provider.open_id_connect_provider_arn,
                {
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                    },
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub": f"repo:{GITHUB_REPO}:*"
                    },
                },
            ),
        )
        # CDK deploys work by assuming the roles created by `cdk bootstrap`.
        role.add_to_policy(iam.PolicyStatement(
            actions=["sts:AssumeRole"],
            resources=[f"arn:aws:iam::{self.account}:role/cdk-*"],
        ))

        CfnOutput(self, "DeployRoleArn", value=role.role_arn)
