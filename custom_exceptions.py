class CustomException(Exception):

    def __init__(self, error_str):
        self.error = error_str
    def __str__(self):
        return str(self.error)

# def tests():
#     try:
#         raise CustomException({'error':'wrong', 'code':402})
#     except CustomException as e:
#         print(e.kwargs)
