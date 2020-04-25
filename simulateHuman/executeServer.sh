#!/bin/sh
sudo hciconfig hci0 piscan 
gnome-terminal \
    --title=SampleServer0 -- bash -c "python3 socketServer.py -u ca84476c-863a-11ea-bc55-0242ac130003 -n SampleServer0; bash" &\
gnome-terminal \
    --title=SampleServer1 -- bash -c "python3 socketServer.py -u e32e38b8-863a-11ea-bc55-0242ac130003 -n SampleServer1; bash" &\
gnome-terminal \
    --title=SampleServer2 -- bash -c "python3 socketServer.py -u 6f1c5660-863c-11ea-bc55-0242ac130003 -n SampleServer2; bash" &\
gnome-terminal \
    --title=SampleServer3 -- bash -c "python3 socketServer.py -u fa3a21ac-863a-11ea-bc55-0242ac130003 -n SampleServer3; bash" &\
gnome-terminal \
    --title=SampleServer4 -- bash -c "python3 socketServer.py -u 00766328-863b-11ea-bc55-0242ac130003 -n SampleServer4; bash" &\
gnome-terminal \
    --title=SampleServer5 -- bash -c "python3 socketServer.py -u 518f38dc-3714-4c58-b2f2-17e9d49c259f -n SampleServer5; bash" &\
gnome-terminal \
    --title=SampleServer6 -- bash -c "python3 socketServer.py -u 65dd7a0b-da85-49bc-a217-6a61f8180857 -n SampleServer6; bash" &\
gnome-terminal \
    --title=SampleServer7 -- bash -c "python3 socketServer.py -u 0b948687-c97b-40bb-b31c-6e5e3e25719a -n SampleServer7; bash" &\
gnome-terminal \
    --title=SampleServer8 -- bash -c "python3 socketServer.py -u 73be0e45-edbe-4e86-8a53-c629baf65272 -n SampleServer8; bash" &\
