import json, sys, io
import base64
import time

from typing import Optional
from fastapi import BackgroundTasks, FastAPI, Header, status, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from predictions import Predict
import registration
from custom_exceptions import CustomException
# from main import Building, GoogleDrive, UserData, return_project_info
from main import Building, UserData, return_project_info

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#--------------- Common Functions ----------------------------#
def print_exception(error, error_code, endpoint, return_dict):
    
    return_dict['msg'] = error
    print(f"Error in {endpoint} endpoint. /n Error: {error}, Code: {error_code}")
    return JSONResponse(content=return_dict, status_code=int(error_code))
#-------------------------------------------------------------#


#--------------- Endpoints -----------------------------------#

@app.get("/index", status_code=200)
def index():

    # a = request.headers.get('a')
    # app.logger.info(a)

    return """<h4> Project Name: Evaluation of Earthquake Resistance of Urban Buildings in\
              Dhaka City using Image Processing and Machine Learning Techniques</h4>\
            <h4> Software Name: Urban Buildings\' Earthquake Resistance Assessor</h4>\
            <h4> Backend API to be used for prediction </h4>
            """


@app.get("/test", status_code=200)
def test():

    # a = request.headers.get('a')
    # app.logger.info(a)
    response.status_code = status.HTTP_201_CREATED
    return jsonify({'success': True})

@app.post("/gen-signup-token", status_code=200)
def gen_signup_token(guestemail: str = Header(...), role: str = Header(...), token: str = Header(...)):
    
    """
    Emails the user a link with a token that can be used to signup to the service.
    """

    return_dict = {'success': False}
    endpoint = 'gen-signup-token'
    try:
        print(f"Executing {endpoint} endpoint")
        registration.Login.validate_jwt_token(token)
        ret_token = registration.SignUp.gen_one_time_token(guestemail, role)
        registration.SignUp.send_signup_email(guestemail, ret_token)

        return_dict['success'], return_dict['token'] = True, ret_token
        return JSONResponse(content=return_dict)

    except CustomException as ce:
        error_code, error = str(ce).split(':')
        return print_exception(error, error_code, endpoint, return_dict)

    except Exception as e:
        return print_exception(str(e), 400, endpoint, return_dict)

@app.post("/signup", status_code=200)
def signup(fullname: str = Header(...), email: str = Header(...), password: str = Header(...), 
           organisation: str = Header(...), designation: str = Header(...), 
           signuptoken: str = Header(...)):
    
    """
    Use this endpoint to signup to the service.
    """

    # TODO Base64 encoding of password
    return_dict = {'success': False}
    endpoint = 'signup'
    try:
        organ, desig, token = organisation, designation, signuptoken
        print(f"Executing {endpoint} endpoint")

        signup_obj = registration.SignUp(fullname, email, password, organ, desig, token)
        role = registration.Login.validate_jwt_token(token)['roles']
        signup_obj.signup(role)

        return_dict['success'], return_dict['msg'] = True, "User account created successfully"
        return JSONResponse(content=return_dict)

    except CustomException as ce:
        error_code, error = str(ce).split(':')
        return print_exception(error, error_code, endpoint, return_dict)

    except Exception as e:
        return print_exception(str(e), 400, endpoint, return_dict)

@app.post("/login", status_code=200)
def login(email: str = Header(...), password: str = Header(...)):
    
    """
    Use this endpoint to login to the service.
    """

    # TODO Base64 encoding of password
    return_dict = {'success': False}
    endpoint = 'login'
    try:
        print(f"Executing {endpoint} endpoint")
        login_obj = registration.Login(email, password)
        login_result = login_obj.validate_credentials()
        role = login_result['role']
        token = login_obj.gen_jwt_token(role)

        return_dict['success'], return_dict['token'], return_dict['role'] = True, token, role
        print(f"Success in {endpoint} endpoint. /n {return_dict}, Code: 200")
        return JSONResponse(content=return_dict)

    except CustomException as ce:
        error_code, error = str(ce).split(':')
        return print_exception(error, error_code, endpoint, return_dict)

    except Exception as e:
        return print_exception(str(e), 400, endpoint, return_dict)

@app.get("/validate_token", status_code=200)
def validate_token(token: str = Header(...)):

    """
    Validate an authorization token.
    """

    return_dict = {'success': False}
    endpoint='validate_token'
    try:
        # if token is None: raise(CustomException('422:Unprocessable Entity'))

        print("Executing {} endpoint".format(endpoint))
        role = registration.Login.validate_jwt_token(token)['roles']

        return_dict['success'], return_dict['role'] = True, role
        return JSONResponse(content=return_dict)

    except CustomException as ce:
        error_code, error = str(ce).split(':')
        return print_exception(error, error_code, endpoint, return_dict)

    except Exception as e:
        return print_exception(str(e), 400, endpoint, return_dict)


@app.get("/get_project_info", status_code=200)
def get_project_info():

    """
    Get all related information about the project.
    """

    return_dict = {'success': False}
    endpoint='get_project_info'
    try:
        print("Executing {} endpoint".format(endpoint))
        proj_info_dict = return_project_info()
        return_dict = {**return_dict, **proj_info_dict}

        return_dict['success'] = True
        return JSONResponse(content=return_dict)

    except CustomException as ce:
        error_code, error = str(ce).split(':')
        return print_exception(error, error_code, endpoint, return_dict)

    except Exception as e:
        return print_exception(str(e), 400, endpoint, return_dict)


@app.post("/predict", status_code=200)
async def predict(background_tasks: BackgroundTasks, buildtype: str= Header(...), area: str= Header(...), 
                  floors: str= Header(...), address: str=Header(...), glass: str= Header(...), 
                  latlong: str= Header(...), img1: bytes = File(...), img_marked: bytes = File(...), 
                  img2: bytes = File(None), img3: bytes = File(None),  img_static: bytes = File(None)):
    
    """
    Predict whether a building requires structural evaluation based on the values and images provided.

    """

    # TODO Take numpy array as input
    return_dict = {'success': False}
    endpoint='predict'
    try:
        print(f"Executing {endpoint} endpoint")
        images = [img1, img2, img3, img_marked, img_static]
        pred_obj = Predict(images, buildtype, floors, latlong, glass, area)
        # background_tasks.add_task(UserData(buildtype, area, floors, glass, address, images))
        # return_dict = pred_obj.calc_fema_score()
        pred_class, pred_scores = pred_obj.pred_class()

        return_dict['success'], return_dict['confidence_score'] = True, pred_scores
        return_dict['struct_eval'] = 'Required' if pred_class else 'Not Required'
        return JSONResponse(content=return_dict)

    except CustomException as ce:
        error_code, error = str(ce).split(':')
        return print_exception(error, error_code, endpoint, return_dict)

    except Exception as e:
        return print_exception(str(e), 400, endpoint, return_dict)

@app.get("/build-info", status_code=200)
def build_info(keys: str= Header(...), keyvalues: str= Header(...), maxbuilds: str= Header(...),
               token: str= Header(...)):
    # TODO get either address or longitude or longitude of building
    # TODO return all information as well as risk assessment
    # TODO return tips to make better
    
    """
    Return list of buildings and their values based on the keys used.
    
    """

    endpoint='build-info'
    return_dict = {'success': False}
    try:
        print("Executing {} endpoint".format(endpoint))
        role = registration.Login.validate_jwt_token(token)['roles']
        keys, keyvalues = keys.split('|'), keyvalues.split('|')
        if len(keys) == 0 or len(keyvalues) == 0: raise Exception('No query key found')
        build_obj = Building(role, keys[1:], keyvalues[1:], maxbuilds)
        
        return_dict['result'] = build_obj.retrieve_from_db()
        return_dict['success'] = True
        return JSONResponse(content=return_dict)

    except CustomException as ce:
        error_code, error = str(ce).split(':')
        return print_exception(error, error_code, endpoint, return_dict)

    except Exception as e:
        return print_exception(str(e), 400, endpoint, return_dict)

@app.get("/download-data", status_code=200, summary="Download data of a folder from Google Drive")
async def download_data(background_tasks: BackgroundTasks, token: str = Header(...), folder: str = Header(...)):
    
    """
    Download data of a particular folder from Google Drive:

    - **token**: token to authorize access to user
    - **folder**: name of the folder to download 
    """
    endpoint='download-data'
    return_dict = {'success': False}
    try:
        print("Executing {} endpoint".format(endpoint))
        role = registration.Login.validate_jwt_token(token)['roles']
        
        with open('folders_to_download.txt', 'a') as f:
            f.write(folder + '\n')
        # def download_from_drive(f_name): GoogleDrive(f_name).get_folder_names()
        # background_tasks.add_task(download_from_drive(folder.lower()))
        return_dict['success'], return_dict['msg'] = True, f'Downloading of folder {folder} started'
        return JSONResponse(content=return_dict)

    except CustomException as ce:
        error_code, error = str(ce).split(':')
        return print_exception(error, error_code, endpoint, return_dict)

    except Exception as e:
        return print_exception(str(e), 400, endpoint, return_dict)


@app.get("/pass-reset-link", status_code=200)
async def pass_reset_link(email: str = Header(...)):

    """
    Email the user a link that they can use to reset the password.
    
    """

    endpoint='pass-reset-link'
    return_dict = {'success': False}
    try:
        print("Executing {} endpoint".format(endpoint))
        role = registration.Login.validate_user(email)['role']
        ret_token = registration.SignUp.gen_one_time_token(email, role)
        registration.Login.send_pass_reset_email(email, ret_token)
        
        return_dict['success'] = True
        return JSONResponse(content=return_dict)

    except CustomException as ce:
        error_code, error = str(ce).split(':')
        return print_exception(error, error_code, endpoint, return_dict)

    except Exception as e:
        return print_exception(str(e), 400, endpoint, return_dict)


@app.get("/pass-reset", status_code=200)
async def pass_reset(token: str = Header(...), email: str = Header(...), newpass: str = Header(...)):

    """
    Call this endpoint to acutally reset the link.
    """

    endpoint='pass-reset'
    return_dict = {'success': False}
    try:
        print("Executing {} endpoint".format(endpoint))
        registration.Login.validate_jwt_token(token)
        registration.Login.pass_reset(email, newpass)
        
        return_dict['success'] = True
        return JSONResponse(content=return_dict)

    except CustomException as ce:
        error_code, error = str(ce).split(':')
        return print_exception(error, error_code, endpoint, return_dict)

    except Exception as e:
        return print_exception(str(e), 400, endpoint, return_dict)

@app.get("/bg_task")
async def root(background_tasks: BackgroundTasks):
    background_tasks.add_task(test_async)
    time.sleep(5)
    return {"message": "Hello World"}


def test_async():
    time.sleep(40)
    

if __name__ == '__main__':
    # uvicorn.run(app, debug='true')
    uvicorn.run(app, host='0.0.0.0', port=5000, debug='true')