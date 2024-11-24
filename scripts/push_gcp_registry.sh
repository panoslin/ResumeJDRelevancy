PROJECT_ID=resumejdrelevancy
REGION=us-central1
REGISTRY_NAME=resume-jd-relevancy-docker-repo
IMAGE_NAME=$REGION-docker.pkg.dev/$PROJECT_ID/$REGISTRY_NAME/resume-jd-relevancy:latest
docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME