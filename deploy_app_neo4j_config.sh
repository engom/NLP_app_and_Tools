#!/bin/bash

# Made to set up ubuntu ec2 instance for konvo applications deployment!

# to install neo4j on ubuntu
# sudo curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/neo4j.gpg
# sudo echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable 4.3" | sudo tee -a /etc/apt/sources.list.d/neo4j.list

# to update instance newly lunched
sudo apt update
sudo apt upgrade

# to install neo4j
# sudo apt install neo4j -y

# to enable and start neo4j service
# sudo systemctl enable neo4j.service
# sudo systemctl start neo4j.service
# sudo systemctl status neo4j.service

# to clone the app github repository
# sudo git clone https://github.com/engom/nlp-app.git

# to install pip & virtualenv python modules
sudo apt install python3-pip

# sudo python3 -m pip install virtualenv
# sudo python3 -m virtualenv env_app

# to activate the virtual env created
# source env_app/bin/activate

# to install app_v2 requirements on env_app
# sudo pip install -r requirements.txt

# to install awscli for s3 access
sudo apt update
sudo apt install awscli -y

# to copy models weights from s3 bucket like this :
# aws s3 cp s3://my_bucket/my_folder/my_file.ext my_copied_file.ext

sudo aws s3 cp s3://konvo-models-store/sentimental_model_cmbert.pth ./sentimental_model_cmbert.pth
sudo aws s3 cp s3://konvo-models-store/emotional_model_cmbert.pth ./emotional_model_cmbert.pth

# to install tmux for background terminal run
sudo apt update
sudo apt install tmux

####################### DO THIS AFTER ##########################
#                                                              #
# Configure neo4j manually:                                    #
#                                                              #
#   i) open config file with nano or vim                       #
#       >> sudo nano /etc/neo4j/neo4j.conf                     #
#   ii) uncomment 2 lines for ports mapping and add 0.0.0.0    #
#        #dbms.listen_address=0.0.0.0:7687                     #
#        #dbms.listen_address=0.0.0.0:7474                     #
#   iii) restart neo4j service                                 #
#       >> sudo systemctl restart neo4j.service                #
#       >> sudo systemctl status neo4j.service                 #
#                                                              #
# Connect to neo4j DATABASE by typing:                         #
#   >> cypher-shell                                            #
#         or                                                   #
#   >> cypher-shell -a 'neo4j://your_hostname:7687'            #
################################################################
