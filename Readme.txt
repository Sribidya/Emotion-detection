1) Clone this repository:
        git clone https://github.com/Sribidya/Emotion-detection.git

2) Install python
       Download and install the latest version of Python from https://www.python.org/downloads/.

RUN FROM THE COMMAND PROMPT

1) Navigate to the project directory:
        cd  path/to/your/project/Emotion-detection

2) Virtual Environment Creation 
		python -m venv myenv         # Create a virtual environment (replace 'myenv'  with your preferred   name)
		source myenv/bin/activate    # Activate virtual environment (for Mac/Linux)
		myenv\Scripts\activate       # Activate virtual environment (for Windows)

3) Install dependencies
		After activation, you can install the required packages using pip.
		pip install <package_name>
		You can also install all dependencies at once:
		pip install -r requirements.txt

4) Running the Python Application
		Navigate to the directory where the .py file is located using the cd command
		cd path/to/your/project
		Run the Python script
		python your_script.py
		
RUN FROM IDE 
1) Install PyCharm from https://www.jetbrains.com/pycharm/download

2) Configure the Python Interpreter
		i) Open Settings:
		Go to File > Settings (Windows/Linux) or PyCharm > Preferences (macOS).
		Navigate to Project: [Project Name] > Python Interpreter.

		ii) Create a Virtual Environment:
		Click the gear icon in the top-right corner and select Add Local Interpreter..

		iii) Choose Virtualenv Environment.Ensure the New Environment is selected and specify the location (default is recommended).
		Select the base Python version installed on your system.

		iv) Set Interpreter:
		After creating the virtual environment, ensure it is selected as the Project Interpreter.

3) Install Dependencies from requirements.txt
		i)  Open the Terminal tab in PyCharm (bottom of the IDE).
		ii) Run the following command to install dependencies
		    pip install -r requirements.txt
		
4) Click the Run button or press Shift + F10 to start the application.