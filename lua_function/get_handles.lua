

------------------------------------------ get handle ------------------------------------

get_joint_hds=function()
    local hd1_1=simGetObjectHandle('rw_joint1')
    local hd1_2=simGetObjectHandle('rw_joint1#0')
    local hd1_3=simGetObjectHandle('rw_joint1#1')
    local hd1_4=simGetObjectHandle('rw_joint1#2')

    local hd2_1=simGetObjectHandle('rw_joint2')
    local hd2_2=simGetObjectHandle('rw_joint2#0')
    local hd2_3=simGetObjectHandle('rw_joint2#1')
    local hd2_4=simGetObjectHandle('rw_joint2#2')

    local hd3_1=simGetObjectHandle('rw_joint3')
    local hd3_2=simGetObjectHandle('rw_joint3#0')
    local hd3_3=simGetObjectHandle('rw_joint3#1')
    local hd3_4=simGetObjectHandle('rw_joint3#2')

    local hd4_1=simGetObjectHandle('rw_joint4')
    local hd4_2=simGetObjectHandle('rw_joint4#0')
    local hd4_3=simGetObjectHandle('rw_joint4#1')
    local hd4_4=simGetObjectHandle('rw_joint4#2')

    local hd5_1=simGetObjectHandle('rw_freeJoint')
    local hd5_2=simGetObjectHandle('rw_freeJoint#0')
    local hd5_3=simGetObjectHandle('rw_freeJoint#1')
    local hd5_4=simGetObjectHandle('rw_freeJoint#2')

    joint_hds={hd1_1,hd1_2,hd1_3,hd1_4, hd2_1,hd2_2,hd2_3,hd2_4, hd3_1,hd3_2,hd3_3,hd3_4, hd4_1,hd4_2,hd4_3,hd4_4, hd5_1,hd5_2,hd5_3,hd5_4}
    return joint_hds
end

get_leg_pan_hds=function()
    local hds={}
    hds[1]=simGetObjectHandle('rw_joint1')
    hds[2]=simGetObjectHandle('rw_joint1#0')
    hds[3]=simGetObjectHandle('rw_joint1#1')
    hds[4]=simGetObjectHandle('rw_joint1#2')
    return hds
end

get_foot_tip_hds=function()
    local hds={}
    hds[1]=simGetObjectHandle('rw_footTip')
    hds[2]=simGetObjectHandle('rw_footTip#0')
    hds[3]=simGetObjectHandle('rw_footTip#1')
    hds[4]=simGetObjectHandle('rw_footTip#2')
    return hds
end

get_foot_target_hds=function()
    local hds={}
    hds[1]=simGetObjectHandle('rw_footTarget')
    hds[2]=simGetObjectHandle('rw_footTarget#0')
    hds[3]=simGetObjectHandle('rw_footTarget#1')
    hds[4]=simGetObjectHandle('rw_footTarget#2')
    return hds
end

get_leg_hds=function(leg_index)
    local hds={}
    if leg_index==1 then
        hds[1]=simGetObjectHandle('rw_joint1')
        hds[2]=simGetObjectHandle('rw_joint2')
        hds[3]=simGetObjectHandle('rw_joint3')
        hds[4]=simGetObjectHandle('rw_joint4')
    elseif leg_index==2 then
        hds[1]=simGetObjectHandle('rw_joint1#0')
        hds[2]=simGetObjectHandle('rw_joint2#0')
        hds[3]=simGetObjectHandle('rw_joint3#0')
        hds[4]=simGetObjectHandle('rw_joint4#0')    
    elseif leg_index==3 then
        hds[1]=simGetObjectHandle('rw_joint1#1')
        hds[2]=simGetObjectHandle('rw_joint2#1')
        hds[3]=simGetObjectHandle('rw_joint3#1')
        hds[4]=simGetObjectHandle('rw_joint4#1') 
    elseif leg_index==4 then
        hds[1]=simGetObjectHandle('rw_joint1#2')
        hds[2]=simGetObjectHandle('rw_joint2#2')
        hds[3]=simGetObjectHandle('rw_joint3#2')
        hds[4]=simGetObjectHandle('rw_joint4#2')  
    end
    return hds
end

get_ik_handles=function()
    local ikgroups={}
    ikgroups[1]=simGetIkGroupHandle('rw_foot')
    ikgroups[2]=simGetIkGroupHandle('rw_foot#0')
    ikgroups[3]=simGetIkGroupHandle('rw_foot#1')
    ikgroups[4]=simGetIkGroupHandle('rw_foot#2')

    return ikgroups
end
------------------------------------------------------
