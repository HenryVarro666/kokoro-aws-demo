#!/usr/bin/env python3
import aws_cdk as cdk

from infra.infra_stack import InfraStack
from infra.oidc_stack import OidcStack

app = cdk.App()

# Phase 3 is opt-in so Phase 1 deploys stay minimal:
#   cdk deploy InfraStack                                   # phase 1
#   cdk deploy --context avatar=true --context avatar_ami=ami-XXX --all   # phase 3
avatar = None
if app.node.try_get_context("avatar") == "true":
    from infra.avatar_stack import AvatarStack
    avatar = AvatarStack(app, "AvatarStack",
                         ami_id=app.node.try_get_context("avatar_ami"))

InfraStack(app, "InfraStack", avatar=avatar)
OidcStack(app, "OidcStack")
app.synth()
