from networktables import NetworkTables
import json

ROBOT_IP_ADDRESS = "10.37.56.2"
NETWORK_TABLE_NAME = "AIPipeline"
DATA_ENTRY_NAME = "data"

NetworkTables.initialize(server=ROBOT_IP_ADDRESS)
table = NetworkTables.getTable(NETWORK_TABLE_NAME)

def post_to_network_tables(data):
    json_data = json.dumps(data)
    table.putString(DATA_ENTRY_NAME, json_data)