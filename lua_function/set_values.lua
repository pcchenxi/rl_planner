
------------------------------------------ set values ------------------------------------
package.path=package.path .. ";/home/xi/workspace/v-rep_code/lua_functions/common/?.lua"
require("common_functions")

set_joint_positions=function(joint_hds, positions)
    for i=1,#positions,1 do
        simSetJointPosition(joint_hds[i], positions[i])
    end
end

set_robot_body=function(robot_hd, pos, ori)
    simSetObjectPosition(robot_hd,-1,pos)
    simSetObjectQuaternion(robot_hd,-1,ori)
end

set_robot_pose=function(robot_hd, joint_hds, state)
    if #state ~= 19 then
        print('in Set Robot Pose: wrong state size! '..#state)
    end

    local robot_pos={}
    local robot_ori={}
    local joint_values={}

    for i=1, 3, 1 do 
        robot_pos[i] = state[i]
    end 

    local index = #robot_pos
    for i=1, 4, 1 do
        robot_ori[i] = state[index + i]
    end
    
    index = #robot_pos + #robot_ori
    for i=1, 12, 1 do
        joint_values[i] = state[index + i]
    end 

    -- print_table(joint_values)

    simSetObjectPosition(robot_hd,-1,robot_pos)
    simSetObjectQuaternion(robot_hd,-1,robot_ori)
    set_joint_positions(joint_hds, joint_values)
end