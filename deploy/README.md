# AWS ECS/Fargate deployment template

`ecs-task-definition.json` is a starting task definition for the API and
dashboard containers. Before registering it:

1. Build and push `Dockerfile.api` and `Dockerfile.dashboard` to ECR.
2. Replace the `${AWS_ACCOUNT_ID}`, `${AWS_REGION}`, `${IMAGE_TAG}`,
   `${TASK_EXECUTION_ROLE_ARN}`, and `${TASK_ROLE_ARN}` placeholders.
3. Provision a VPC, private subnets, security groups, an Application Load
   Balancer, and an ECS cluster/service. Expose only the dashboard and API
   routes that your users need.
4. Put logs in CloudWatch and configure health checks for both containers.

The task expects model files to be included in the container image. Do not
commit generated artifacts containing sensitive data; publish reviewed model
artifacts through your organization's controlled release process. The sample
resources are a deployment template, not a production security review.
