import argparse
import requests
import time
from azureml.core import Workspace
from azureml.core.webservice import AksWebservice, AciWebservice
from ml_service.util.env_variables import Env
import secrets


input = {"data": [[0.9881476548697458,
    1.0574118809171713,
    1.107516482772424,
    1.1680586851515704,
    1.1706494289622553,
    1.1173641094266091,
    1.1230256133841114,
    1.1214253211011833,
    1.1570478978922298,
    1.0958661153419724,
    0.8962792845174279,
    0.9837183386596208,
    1.0047096455536586,
    1.118186522268419,
    1.0535245412580314,
    0.9287346565314413,
    0.9301213623382195,
    0.9832277974387773,
    0.8848283088742313,
    0.9741439267336769,
    0.7272957025494287,
    0.8690971745069971,
    0.848041469004961,
    0.7408090254185772,
    0.7196108666182776],
    [1.039451917715693,
    1.0522419247963042,
    1.1090394053790862,
    1.1030840957879928,
    1.1362953360901118,
    1.1737614469976543,
    1.0585470469401108,
    1.0637206436139601,
    1.0873102142916191,
    1.01837173608471,
    0.9667168488572627,
    1.075516969333446,
    1.0689746668927909,
    1.0810641891083232,
    1.0342815311577007,
    1.0376777233997976,
    0.9497229075103204,
    0.9454843103384318,
    0.9969723153192604,
    0.9415068000721494,
    0.7258200594488118,
    0.7853886057993892,
    0.8315852457929973,
    0.7718692332905781,
    0.754231641862637]]}
output_len = 2


def call_web_service(e, service_type, service_name):
    aml_workspace = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group
    )
    print("Fetching service")
    headers = {}
    if service_type == "ACI":
        service = AciWebservice(aml_workspace, service_name)
    else:
        service = AksWebservice(aml_workspace, service_name)
    if service.auth_enabled:
        service_keys = service.get_keys()
        headers['Authorization'] = 'Bearer ' + service_keys[0]
    print("Testing service")
    print(". url: %s" % service.scoring_uri)
    output = call_web_app(service.scoring_uri, headers)

    return output
    

def call_web_app(url, headers):

    # Generate an HTTP 'traceparent' distributed tracing header
    # (per the W3C Trace Context proposed specification).
    headers['traceparent'] = "00-{0}-{1}-00".format(
        secrets.token_hex(16), secrets.token_hex(8))

    retries = 600
    for i in range(retries):
        try:
            response = requests.post(
                url, json=input, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if i == retries - 1:
                raise e
            print(e)
            print("Retrying...")
            time.sleep(1)


def main():

    parser = argparse.ArgumentParser("smoke_test_scoring_service.py")

    parser.add_argument(
        "--type",
        type=str,
        choices=["AKS", "ACI", "Webapp"],
        required=True,
        help="type of service"
    )
    parser.add_argument(
        "--service",
        type=str,
        required=True,
        help="Name of the image to test"
    )
    args = parser.parse_args()

    e = Env()
    if args.type == "Webapp":
        output = call_web_app(args.service, {})
    else:
        output = call_web_service(e, args.type, args.service)
    print("Verifying service output")

    assert "result" in output
    assert len(output["result"]) == output_len
    print("Smoke test successful.")


if __name__ == '__main__':
    main()
