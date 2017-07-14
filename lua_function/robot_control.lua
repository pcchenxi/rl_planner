-- action: vx, vy, vw, vl
package.path=package.path .. ";/home/xi/workspace/v-rep_code/rl_planner/lua_functions/?.lua"
require("get_values")
require("set_values")
require("get_handles")

step = 0.05
dx = step
dy = step
dw = step
dh = step/3.0
dl = step

collision_hd_1 = simGetCollectionHandle('robot_body')
collision_hd_2 = simGetCollectionHandle('obstacle_all')


function do_action(robot_hd, joint_hds, action)
    if(#action == 1) then
        action = action_list[action[1]]
    end

    local current_pos=simGetObjectPosition(robot_hd,-1)
    local current_ori=simGetObjectQuaternion(robot_hd,-1)
    local current_joints = get_joint_positions(joint_hds)
    local h, l = get_current_pose(robot_hd, joint_hds)

    local sample_pose = {}
    sample_pose[1] = dx*action[1]
    sample_pose[2] = dy*action[2] 
    sample_pose[3] = dh*action[4] 

    local sample_ori = {}
    sample_ori[1] = current_ori[1] 
    sample_ori[2] = current_ori[2]   
    sample_ori[3] = current_ori[3] + dw*action[3] 
    sample_ori[4] = current_ori[4]

    print (sample_ori[3])
    simSetObjectPosition(robot_hd,robot_hd,sample_pose)
    simSetObjectQuaternion(robot_hd,-1,sample_ori)

    local r0 = 0.07238 --math.sqrt(knee_pos[1]^2 + knee_pos[2]^2)
    local r1 = 0.10545 --math.sqrt(tip_pos[1]^2 + tip_pos[2]^2)

    local tilt_pos = {}
    local foot_pos = {}

    -- sample feet and joint --
    for i=1, 4, 1 do
        local leg_joints=get_leg_hds(i) -- pan, tilt, knee, ankle

        tilt_pos = simGetObjectPosition(leg_joints[2], -1)
        foot_pos = simGetObjectPosition(leg_joints[4], leg_joints[2])

        foot_pos[1] = foot_pos[1] + dl*action[5] 
        foot_pos[2] = tilt_pos[3] - 0.0444

        if foot_pos[2] < 0 then
            displayInfo('too low '..i..' '..foot_pos[1]..' '..foot_pos[2] )
            restore_pose(robot_hd, joint_hds, current_pos, current_ori, current_joints)
            return {h, l}, 'f'
        end

        local knee_x, knee_y = get_intersection_point(0, 0, foot_pos[1], foot_pos[2], r0, r1)

        if knee_x == -1 or knee_x ~= knee_x then
            displayInfo('no pose found '..i..' '..foot_pos[1]..' '..foot_pos[2] )
            restore_pose(robot_hd, joint_hds, current_pos, current_ori, current_joints)
            return {h, l}, 'f'
        end

        local foot_x_fromknee = foot_pos[1]-knee_x
        local foot_y_fromknee = foot_pos[2]-knee_y

        -- displayInfo('knee pos: '..foot_x_fromknee..' '..foot_y_fromknee)

        local angle_thigh = math.atan(knee_y/knee_x)
        local angle_knee = math.atan(foot_y_fromknee/foot_x_fromknee)
        if angle_knee<0 then 
            angle_knee = angle_knee + math.pi 
        end

        simSetJointPosition(leg_joints[2], angle_thigh)
        simSetJointPosition(leg_joints[3], angle_knee-angle_thigh)

        print (i, angle_thigh, angle_knee-angle_thigh)
        -- local real_foot_pos = 
    end

    -- check collision --
    if is_valid() then 
        return {sample_pose[3], math.abs(foot_pos[1])}, 't'
    else
        restore_pose(robot_hd, joint_hds, current_pos, current_ori, current_joints)
        -- displayInfo('collide '..i..' '..foot_pos[1]..' '..foot_pos[2] )
        return {h, l}, 'f'      
    end
end

function restore_pose(robot_hd, joint_hds, poi, ori, joints)
    simSetObjectPosition(robot_hd,-1,poi)
    simSetObjectQuaternion(robot_hd,-1,ori)
    set_joint_positions(joint_hds, joints)
end

function is_valid()
    local is_valid = true
    local res=simCheckCollision(collision_hd_1, collision_hd_2)

    if res > 0 then 
        is_valid = false
    end

    return is_valid
end

function get_current_pose(robot_hd, joint_hds)
    local leg_hd = get_leg_hds(1)
    local current_pos=simGetObjectPosition(robot_hd,-1)
    local foot_pos = simGetObjectPosition(leg_hd[4], leg_hd[2])

    return current_pos[3], math.abs(foot_pos[1])
end

set_collision_hd=function(name1, name2)
    _collision_hd_1 = simGetCollectionHandle(name1)
    _collision_hd_2 = simGetCollectionHandle(name2)
end

