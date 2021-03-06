FDS_SCRIPT_TEST_1 = '''
************************************************************************************************************************
FDS Script Descrption
=====================
This is a sythetic testing FDS script which consists of predefined commands and this will be check against the results
came out from the functions:

`fdspy.lib.fds_script_proc_analyser: fds_analyser_general`
`fdspy.lib.fds_script_proc_analyser: fds_analyser_mesh`
`fdspy.lib.fds_script_proc_analyser: fds_analyser_slcf`
`fdspy.lib.fds_script_proc_analyser: fds_analyser_hrr`
`fdspy.lib.fds_script_proc_analyser: fds_analyser_hrr_fig`

The prescribed FDS commands:

commmands count:                    27
sim. duration:                      300

mesh count:                         3
cell count:                         7500
ave. cell size:                     0.1
mesh 1 (cell size, cell count):     0.1, 2500
mesh 2 (cell size, cell count):     0.1, 2500
mesh 3 (cell size, cell count):     0.1, 2500

SLCF count:                         6
SLCF 'VISIBILITY' count:            3
SLCF 'TEMPERATURE' count:           2
SLCF 'VELOCITY' count:              1

Miscellaneous
=============
Lasted updated:     11 Nov 2019
Author:             Yan FU
FDS version:        6.7.1
************************************************************************************************************************

&HEAD CHID='test_case_1'/
&TIME T_END=720/
&DUMP COLUMN_DUMP_LIMIT=.TRUE., DT_SL3D=0.25/
&MISC VISIBILITY_FACTOR=8.0/

&MESH IJK=50,50,10, XB=-2.5,2.5,-2.5,2.5,0,1/
&MESH IJK=50,50,10, XB=-2.5,2.5,-2.5,2.5,1,2/
&MESH IJK=50,50,10, XB=-2.5,2.5,-2.5,2.5,2,3/

&REAC ID='POLYURETHANE', FYI='NFPA Babrauskas', FUEL='REAC_FUEL', C=6.3, H=7.1, O=2.1, N=1.0, SOOT_YIELD=0.07, 
      HEAT_OF_COMBUSTION=2.0E4/

&PROP ID='Default', QUANTITY='LINK TEMPERATURE', ACTIVATION_TEMPERATURE=74.0, RTI=80.0/

&DEVC ID='HD', PROP_ID='Default', XYZ=0.0,0.0,0.5/
&DEVC ID='HD', PROP_ID='Default', XYZ=0.0,0.0,1.5/
&DEVC ID='HD', PROP_ID='Default', XYZ=0.0,0.0,2.5/

&SURF ID='surf_fire', COLOR='RED', HRRPUA=500.0, TAU_Q=-300.0/

&OBST ID='obst_fire', XB=-0.5,0.5,-1.0,1.0,0.1,0.2, SURF_IDS='surf_fire'/ 
&OBST ID='floor', XB=-2.5,2.5,-2.5,2.5,0.0,1.0, RGB=255,51,51, SURF_ID='INERT'/ 
&OBST ID='wall', XB=-2.5,2.5,2.4,2.5,0.0,3.0, RGB=255,51,51, SURF_ID='INERT'/ 

&HOLE ID='Hole', XB=-1.0,1.0,2.3,2.6,1.0,2.0 COLOR='INVISIBLE'/ 

&SLCF QUANTITY='VISIBILITY', PBX=0.0/
&SLCF QUANTITY='VISIBILITY', PBY=0.0/
&SLCF QUANTITY='VISIBILITY', PBZ=2.0/
&SLCF QUANTITY='VISIBILITY', PBX=1.0/
&SLCF QUANTITY='VISIBILITY', PBY=1.0/
&SLCF QUANTITY='VISIBILITY', PBZ=3.0/
&SLCF QUANTITY='TEMPERATURE', PBX=0.0/
&SLCF QUANTITY='TEMPERATURE', PBY=0.0/
&SLCF QUANTITY='VELOCITY', PBX=0.0/

&TAIL /'''