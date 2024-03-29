import os
from onshape.client import Client
from get_intermediate_shapes import get_intermediate_shapes, call
from synthetic_cad_utils import get_object

def create_new_document(client):
    # Create a new document
    document = client.new_document(name="My New Document", owner_type=0, public=False)
    did = document.json()["id"]
    print(f"New document created with ID: {did}")
    return did

def get_default_workspace(client, did):
    # Get the default workspace ID
    document_details = client.get_document(did)
    wid = document_details.json()["defaultWorkspace"]["id"]
    print(f"Default workspace ID: {wid}")
    return wid

def create_new_element(client, did, wid):
    # Create a new element in the document
    response = client.create_assembly(did, wid, name="My New Assembly")
    eid = response.json()["id"]
    print(f"New element created with ID: {eid}")
    return eid

def define_sequence_and_get_object(client, url):
    # Define the sequence of operations
    seq_template = [
        {"op": "SKETCH", "shape": "SQUARE", "size": 1.0},
        {"op": "EXTRUDE", "depth": 1.0}
    ]
    
    # Call the get_object function with the sequence and url
    get_object(seq_template, url)


if __name__ == "__main__":
    # Set up the Onshape client
    base_url = "https://cad.onshape.com"
    client = Client(stack=base_url, logging=True)
    
    # Create a new document
    # did = create_new_document(client)
    
    # # Get the default workspace ID
    # wid = get_default_workspace(client, did)
    
    # # Create a new element in the document
    # eid = create_new_element(client, did, wid)
    
    # url = f"{base_url}/documents/{did}/w/{wid}/e/{eid}"
    
    define_sequence_and_get_object(client, base_url)

    print("Onshape project created successfully!")