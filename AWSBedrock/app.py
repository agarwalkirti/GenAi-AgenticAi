# conda deactivate as our venv already have boto3 installed
# pip install boto3 -t python/ -> latest boto will be installed in python folder and we will upload in lambda function portal aws as boto_layer.zip
# Add package in form of layer
# layer addition-(boto3_layer) , api gateway and s3 bucket will be created in aws console
# BlogGenerationApi : https://ffue409wu6.execute-api.us-east-1.amazonaws.com/dev
# awsbedrockbloggenerationbucket  : s3 bukcet unique name
# lambda function name:awsappbedrock
# role name: awsappbedrock-role-9v379mc1 
# cloudwatch used to debug errors and see our logs
# formatted_prompt was confusing llm hence using simple prompt

import boto3
import botocore.config
import json
from botocore.exceptions import ClientError
from datetime import datetime

def blog_generate_using_bedrock(blogtopic:str)->str:
    # Define the prompt for the model. Human:Write a 200 words blog on the topic {blogtopic}. Assistant:
    # Embed the prompt in Llama 3's instruction format.
    prompt = f"""Human: Write a 200 words blog on the topic "{blogtopic}".
                Assistant:"""
    formatted_prompt = f"""
    "<|begin_of_text|>
    <|start_header_id|>
    Human:
    <|end_header_id|>
    \n Write a 200 words blog on the topic {blogtopic}.\n
    <|eot_id|>\n
    <|start_header_id|>
    Assistant:
    <|end_header_id|>\n"
    """
    model_id = "meta.llama3-8b-instruct-v1:0"

    body={
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.7,
            "top_p": 0.9
        }
    
    try:
        bedrock = boto3.client("bedrock-runtime",region_name="us-east-1",
                               config=botocore.config.Config(
                                   read_timeout=300,retries={'max_attempts': 3}
                                   )
                               )
        response = bedrock.invoke_model(
                                        modelId=model_id,
                                        contentType="application/json",
                                        accept="application/json",
                                        body=json.dumps(body)
                                    )
        #response_content = response.get('body').read()
        response_content = response["body"].read().decode("utf-8")
        response_data = json.loads(response_content)
        print("Raw response:", response_data)

        # Some models return "generation", others "outputs"
        if "generation" in response_data:
            blog_details = response_data["generation"]
        elif "outputs" in response_data:
            blog_details = response_data["outputs"][0]["text"]
        else:
            blog_details = str(response_data)
        return blog_details
    
    except(ClientError, Exception) as e:
        print(f"ERROR: Error generating the blog.Can't invoke '{model_id}'. Reason: {e}")
        #exit(1)
        return ""

def save_blog_details_s3(s3_key,s3_bucket,generate_blog):
    s3=boto3.client('s3')
    try:
        s3.put_object(Bucket = s3_bucket,Key = s3_key,Body = generate_blog)
        print("Blog saved to s3")
    except Exception as e:
        print("Error when saving the blog to s3. Reason: {e}")


# blog topic will be captured in event variable,after blog is generated we will save blog in s3 bucket in form of txt file
def lambda_handler(event, context):
    event = json.loads(event['body'])
    blogtopic = event['blog_topic'] #passsed blog_topic from postman

    generate_blog = blog_generate_using_bedrock(blogtopic=blogtopic)

    if generate_blog:
        print("Blog was generated. Going to save in s3.")
        current_time = datetime.now().strftime('%H%M%S') #hours:months:seconds format
        s3_key = f"blog-output/{current_time}.txt"
        s3_bucket = 'awsbedrockbloggenerationbucket'
        save_blog_details_s3(s3_key,s3_bucket,generate_blog)

    else:
        print("No blog was generated")

    return {
        'statusCode': 200,
        'body': json.dumps('Blog Generation is completed and blog saved!')
    }


     