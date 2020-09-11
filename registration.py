from time import time
import sys
from os.path import abspath, dirname, join
import copy
import configparser
import smtplib, ssl
import email.message

import jwt

cur_dir = abspath(__file__)
par_dir = dirname(dirname(cur_dir))
sys.path.insert(0, par_dir)

import db_interface as db
from custom_exceptions import CustomException

config_file = 'config.ini'
config = configparser.ConfigParser()                                     
config.read(config_file)

SECRET_KEY = config.get('REGISTRATION', 'SECRET_KEY')
JWT_EXPIRY = config.get('REGISTRATION', 'JWT_EXPIRY')
USR_COL = config.get('REGISTRATION', 'USER_COLLECTION')
EMAIL = config.get('EMAIL', 'EMAIL')
PASSWORD = config.get('EMAIL', 'PASSWORD')
SIGNUP_URL = config.get('SIGNUP', 'SIGNUP_URL')
PASS_RESET_URL= config.get('SIGNUP', 'PASS_RESET_URL')


"""User Levels/Roles
1) Admin: Access to all resources and API endpoints. Can send invitation to users and moderators.
2) Moderator: Can view information of all buildings.
3) User: Can view basic information and those relating to the user's building.

"""

# TODO Add encryption to password saving

class Login:
    
    def __init__(self, email, password):
        self.email = email
        # self.password = base64.b64decode(password.encode('utf-8')).decode('utf-8')
        self.password = password

    def validate_credentials(self):
        
        try:
            db_obj = db.DatabaseInterface(USR_COL)               # Initialize with collection/table name
            query = {'email': self.email}    
            
            result = db_obj.get(query)
            if result == False: raise CustomException('404:User not found')
            if result['password'] != self.password: raise CustomException('401:Unauthenticated')
            return result
        
        except CustomException as ce:
            raise CustomException(ce)

        except Exception:
            # log
            raise CustomException("500:Internal Server Error")

    def gen_jwt_token(self, role):
        
        token = jwt.encode({
            'email': self.email,
            'roles': role,
            'exp': time() + float(JWT_EXPIRY),
            'iss': 'UBERA'
        }, SECRET_KEY, algorithm='HS256').decode('utf-8')

        return token
    
    @staticmethod
    def validate_user(email):
        
        db_obj = db.DatabaseInterface(USR_COL)               # Initialize with collection/table name
        query = {'email': email}
        result = db_obj.get(query)
        
        if result == False: raise CustomException('404:User doesn\'t exist')
        return result

    @staticmethod
    def validate_jwt_token(token):
        
        try:
            result = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            return result
        
        except (jwt.exceptions.InvalidTokenError, jwt.exceptions.InvalidSignatureError):
            raise CustomException("401:Invalid Token")
        except jwt.exceptions.ExpiredSignatureError: 
            raise CustomException("401:Expired Token")
        except Exception as e:
            raise CustomException("500:Internal Server Error")

    @staticmethod
    def send_pass_reset_email(receiver, token):
        
        try:
            port = 465  # For SSL
            sender, password = EMAIL, PASSWORD
            pass_reset_url = PASS_RESET_URL + f'/{token}'

            body = f"Dear Sir/Madam,\n\nThis message is sent from Urban Buildings' Earthquake Resistance Assessor(UBERA).\nGo to the URL below to reset your password. The link has a validity of 1 hour.\nPlease access the link within that time.\n\nLink: {pass_reset_url}\n\nBest Regards,\nUBERA Team"
            
            message= email.message.Message()
            message['Subject'] = 'UBERA Password Reset'
            message['From'] = sender
            message['To'] = receiver
            message.set_payload(body)

            # Create a secure SSL context
            context = ssl.create_default_context()  
            with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
                server.login(sender, password)
                server.sendmail(sender, receiver, message.as_string())
        
        except Exception:
            # log
            raise CustomException("500:Mail not sent")
    
    @staticmethod
    def pass_reset(email, new_pass):
        
        try:
            result = validate_user(email)

            new_dict = copy.deepcopy(result)
            new_dict['password'] = new_pass
            
            ret = db_obj.update(result, new_dict)
            if ret == False: raise CustomException('500:Password not updated')
            return result
        
        except CustomException as ce:
            raise CustomException(ce)

        except Exception as e:
            print(e)
            raise CustomException("500:Internal Server Error")

class SignUp:

    def __init__(self, fullname, email, password, organisation, designation, signup_token):
        self.fullname = fullname
        self.email = email
        self.password = password
        self.organisation = organisation
        self.designation = designation
        self.token = signup_token

    @staticmethod
    def check_user(email):

        db_obj = db.DatabaseInterface(USR_COL)               # Initialize with collection/table name
        query = {'email': email}
        result = db_obj.get(query)
        
        if result != False: raise CustomException('403:User already exists')
        return result

    def signup(self, role):
        # Possible roles: user, moderator, admin
        try:
            db_obj = db.DatabaseInterface(USR_COL)               # Initialize with collection/table name
            query = [{'email': self.email, 'password': self.password, 'fullname': self.fullname, 
                      'designation': self.designation, 'organisation':self.organisation,'role': role}]
            
            result = db_obj.insert(query)
            if result == False: raise CustomException('403:User already exists')
            return result
        
        except CustomException as ce:
            raise CustomException(ce)

        except Exception as e:
            print(e)
            raise CustomException("500:Internal Server Error")

    @staticmethod
    def gen_one_time_token(guest_email, role):
    
        token = jwt.encode({
            'guest_email': guest_email,
            'roles': role,
            'exp': time() + 3600.0,                             # Validity of one hour
            'iss': 'UBERA'
        }, SECRET_KEY, algorithm='HS256').decode('utf-8')

        return token

    @staticmethod
    def send_signup_email(receiver, token):
        
        try:
            SignUp.check_user(receiver)
            port = 465  # For SSL
            sender, password = EMAIL, PASSWORD
            signup_url = SIGNUP_URL + f'/{token}'

            body = f"Dear Sir/Madam,\n\nThis message is sent from Urban Buildings' Earthquake Resistance Assessor(UBERA).\nGo to the URL below to set up your account. The link has a validity of 1 hour.\nPlease access the link within that time.\n\nLink: {signup_url}\n\nBest Regards,\nUBERA Team"
            
            message= email.message.Message()
            message['Subject'] = 'UBERA Registration'
            message['From'] = sender
            message['To'] = receiver
            message.set_payload(body)

            # Create a secure SSL context
            context = ssl.create_default_context()  
            with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
                server.login(sender, password)
                server.sendmail(sender, receiver, message.as_string())
        
        except CustomException as ce:
            print(str(ce))
            raise CustomException(ce)

        except Exception as e:
            print(f"SignUp email not sent due to {str(e)}")
            raise CustomException("500:Mail not sent")

def tests():
    email, password = 'ubera.cvis', 'uber'
    obj = Login(email, password)
    print(obj.validate_credentials())

