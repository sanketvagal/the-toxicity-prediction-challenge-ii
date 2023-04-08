# the-toxicity-prediction-challenge-ii

**Name: Sanket Dilip Vagal  
StFX Student ID Number: 202207184  
StFX email ID: x2022fjg@stfx.ca**

To get started, clone this repository locally.

### Steps to execute the Docker image

1. Make sure Docker Daemon is running on your machine

2. Set current working directory to `the-toxicity-prediction-challenge-ii`

3. Build the docker image
`docker build -t the-toxicity-prediction-challenge-ii .`

4. Run the docker image 

	For Unix-based machines (Linux, MacOS):
	`docker run -v ${PWD}/output:/usr/src/app/output -m 13g --cpus=8 -e PYTHONUNBUFFERED=1 the-toxicity-prediction-challenge-ii`

	For Windows machines:
	`docker run -v %cd%/output:/usr/src/app/output -m 13g --cpus=8 -e PYTHONUNBUFFERED=1 the-toxicity-prediction-challenge-ii`.

5. The final submission.csv file will be generated in the `the-toxicity-prediction-challenge-ii/output` folder

### Steps to execute the python script via command line

Building and running the docker image may take long amount of time. Instead, to run the `main.py` script via command line:

1. Install the required dependencies by running the following command:
`pip install --no-cache-dir -r requirements.txt`

2. Run the `main.py` script
`python3 main.py`

3. The final submission.csv file will be generated in the `output` folder of the current working directory