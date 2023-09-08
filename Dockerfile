# Specify the parent image from which we build
FROM stereolabs/zed:3.7-gl-devel-cuda11.4-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

RUN python3 /usr/local/zed/get_python_api.py

# Install packages not available in slim image 
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \ 
#     libglib2.0-0 \
#     sudo \ 
#     libpng-dev \ 
#     libgomp1

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip

# Define environment variable (if needed)
ENV PYTHONPATH "${PYTHONPATH}:/app"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./video_detection/ /app/video_detection/
COPY ./models/ /app/models/

# Run multiple_face_detect_zed.py when the container launches
CMD ["python3", "video_detection/multiple_face_detect_zed.py"]
