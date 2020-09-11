import configparser
from os.path import abspath, dirname, join

from pymongo import MongoClient
client = MongoClient()

cur_dir = abspath(__file__)
par_dir = dirname(dirname(cur_dir))
config_file = 'config.ini'

config = configparser.ConfigParser()                                     
config.read(config_file)

DB = config.get('DB', 'DB_NAME')

class DatabaseInterface():

    def __init__(self, collection):
        self.collection = client[DB][collection]
    
    def get_by_limit(self, query_dict={}, limit=0):
        """Return a limited number of key-value pairs in DB(All collections for MongoDB)."""
        
        coll = self.collection
        response = coll.find(query_dict, limit=limit)
        
        result = []
        for item in response:
            del item['_id']
            result.append(item)
        
        if len(result) == 0:
            result = False 
        return result

    def count(self):
        """Count how many documents match the query."""
        pass

    def get(self, query_dict):
        """query_dict is a dictionary with unique field as key.
           Read first occurence from matching the query from the collection."""

        coll = self.collection
        length = len(query_dict)

        if length == 0: raise Exception("No query value")
        result = coll.find_one(query_dict)                  # Find first occurence

        if result == None or 0:
            result = False 
        else:
            del result['_id']
        return result

    def insert(self, key_value_list):
        """key_value_list is a list of dictionaries. Each element/dict needs to be inserted."""
        
        coll = self.collection
        length = len(key_value_list)
        
        if length == 0:
            raise Exception("No key-value to insert")
        elif length == 1:
            if self.exists(key_value_list[0]) == True: return False
            result = coll.insert_one(key_value_list[0])
        else:
            result = coll.insert_many(key_value_list)
        return result.acknowledged

    def exists(self, query_dict):
        
        coll = self.collection
        result = coll.count_documents(query_dict)
        ret = True if result == 1 else False
        return ret

    def update(self, query_dict, new_dict):
        
        coll = self.collection
        if bool(query_dict) == False: raise Exception("Query dictionary is empty")
        if bool(new_dict) == False: raise Exception("No new value to insert")
        if self.exists(query_dict) == False: return False

        new_dict = {"$set": new_dict}
        result = coll.update_one(query_dict, new_dict)
        return result.acknowledged

    def delete(self, query_dict):
        
        coll = self.collection
        if bool(query_dict) == False: raise Exception("Query dictionary is empty")
        if self.exists(query_dict) == False: return False

        result = coll.delete_one(query_dict)
        return result.acknowledged

def tests():
    db_obj = DatabaseInterface('users')
    ret_result = db_obj.insert([{'username': 'tahsin6'}])
    print(ret_result)


