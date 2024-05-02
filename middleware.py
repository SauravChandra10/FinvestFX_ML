from flask import request
import functools

# middleware -> auth 
def auth(view_func):
    @functools.wraps(view_func)
    def decorated(*args, **kwargs):
        # 1. accept from params
        token = request.args.get("token")

        if not token:
            res={
                "status":False,
                "message":"Error",
                "data":'Missing token'
            }
            return res, 401
        

        # define dummy fucntion which checks for token fom backend and return user object
        try:
            if token != 'toA72nrlQAHlBU7':
                res={
                    "status":False,
                    "message":"Error",
                    "data":'Unautorised user'
                }
                return res, 401

            return view_func(*args, **kwargs)

        except ValueError as e:
            # Handle errors during token verification or backend communication
            res={
                "status":False,
                "message":"Error",
                "data":'Internal server error'
            }
            return res, 500
        
    return decorated