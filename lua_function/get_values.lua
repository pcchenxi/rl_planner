
------------------------------------------ get value ------------------------------------
get_foottip_positions=function(foot_hds)
    local foot_poisitons={}
    for i=1, 4, 1 do
        foot_poisitons[i]=simGetObjectPosition(foot_hds[i], -1)
    end
    return foot_poisitons
end


get_joint_positions=function(joint_hds, num)
    if num == -1 or num == nil then
        num = #joint_hds
    end

    local joint_positions={}
    for i=1, num, 1 do
        joint_positions[i]=simGetJointPosition(joint_hds[i])
    end
    return joint_positions
end


get_robot_pose=function(robot_hd, joint_hds, robot_dim, joint_dim)
    startpos=simGetObjectPosition(robot_hd,-1)
    startorient=simGetObjectQuaternion(robot_hd,-1)
    local joint_pose = get_joint_positions(joint_hds, joint_dim)

    local pose={}
    if robot_dim == 1 then
        pose[1]=startpos[1] -- x
        pose[2]=startpos[2] -- y
    elseif robot_dim == 2 then
        pose[1]=startpos[1] -- x
        pose[2]=startpos[2] -- y
        pose[3]=startorient[3] -- yaw        
    elseif robot_dim == 3 then
        pose[1]=startpos[1] -- x
        pose[2]=startpos[2] -- y
        pose[3]=startpos[3] -- z
    elseif robot_dim == 4 then
        pose[1]=startpos[1] -- x
        pose[2]=startpos[2] -- y
        pose[3]=startpos[3] -- z
        pose[4]=startorient[3] -- yaw
    elseif robot_dim == 6 then
        pose[1]=startpos[1] -- x
        pose[2]=startpos[2] -- y
        pose[3]=startpos[3] -- y
        pose[4]=startorient[1] -- row
        pose[5]=startorient[2] -- pitch
        pose[6]=startorient[3] -- yaw
        pose[7]=startorient[4] -- 
    end

    for i=1, #joint_pose, 1 do
        pose[#pose+1] = joint_pose[i]
    end
    return pose
end