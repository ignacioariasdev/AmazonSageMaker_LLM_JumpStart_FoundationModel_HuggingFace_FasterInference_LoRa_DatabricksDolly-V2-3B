# Import necessary libraries
import json
import boto3
import os
import re
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a SageMaker client
sagemaker_client = boto3.client("sagemaker-runtime")

# Define Lambda function
def lambda_handler(event, context):
    # Log the incoming event in JSON format
    logger.info('Event: %s', json.dumps(event))
    
    # Clean the body of the event: remove excess spaces and newline characters
    cleaned_body = re.sub(r'\s+', ' ', event['body']).replace('\n', '')

    # Log the cleaned body
    logger.info('Cleaned body: %s', cleaned_body)

    # Invoke the SageMaker endpoint with the cleaned body as payload and content type as JSON
    response = sagemaker_client.invoke_endpoint(
        EndpointName=os.environ["ENDPOINT_NAME"], 
        ContentType="application/json", 
        Body=cleaned_body
    )

    # Load the response body and decode it
    result = json.loads(response["Body"].read().decode())

    # Return the result with status code 200 and the necessary headers
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST'
        },
        'body': json.dumps(result)
    }
