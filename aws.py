"""Module to connect to aws ec2 instance"""
import argparse
import boto3

INSTANCE_ID = 'i-0a9b6c892969f7556'

def main():
    """To start/stop ec2 instance"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    ec2 = boto3.client('ec2')
    if args.start:
        ec2.start_instances(InstanceIds=[INSTANCE_ID])
        print("Started EC2 instance")
        #pending->running
    elif args.stop:
        ec2.stop_instances(InstanceIds=[INSTANCE_ID])
        print("Stopped EC2 instance")
        #stopping->stopped
    else:
        response = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
        current_state = (response['Reservations'][0]['Instances'][0]['State']['Name'])
        print("EC2 instance status : ", current_state)
        if current_state == 'running':
            public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
            print("Public IP : ", public_ip)
            print("Make sure Kidaptive-VPN is connected")
            print("Connect to EC2 using :")
            print("ssh {}@{}".format('rraju', public_ip))
        #'stopped'
        #'running'

if __name__ == '__main__':
    main()
