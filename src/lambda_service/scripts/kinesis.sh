export KINESIS_STREAM_INPUT="source_stream-mlops-zoomcamp"
export KINESIS_STREAM_OUTPUT="output_stream-mlops-zoomcamp"

#s3://mlops-zoomcamp-bucket-2025/1/models/m-52d4e09e0bba4ce889c106e5b5089aa2/artifacts/
aws kinesis put-record \
  --stream-name ${KINESIS_STREAM_INPUT} \
  --partition-key 1 \
  --data "$(echo '{"cityname": "Raleigh", "state": "NC", "bedrooms": 3, "bathrooms": 2, "square_feet": 1300}' | base64)"

