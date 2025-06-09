# backend_api

### To install all libs:
pip install -r app/requirements.txt

### Get Containers
curl --location 'http://127.0.0.1:8000/get_containers'

### Insert Container
curl --location 'http://127.0.0.1:8000/insert_container' \
--header 'Content-Type: application/json' \
--data '{
  "image_base64": "data:image/jpeg;base64, {add img base64 here}"
}'

### Remove Container
curl --location --request DELETE 'http://127.0.0.1:8000/remove_container?id=1'