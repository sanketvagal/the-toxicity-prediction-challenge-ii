FROM python:3.8

# set a directory for the app
WORKDIR /usr/src/app

# copy all the files to the container
COPY . .

# create the output directory, if it doesn't exist
RUN test -d output || mkdir output

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# run the command
CMD ["python", "./main.py"]