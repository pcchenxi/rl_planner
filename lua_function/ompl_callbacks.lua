package.path=package.path .. ";/home/xi/workspace/v-rep_code/rl_planner/lua_functions/?.lua"

require("get_values")
require("set_values")
require("get_handles")

local torch = require 'torch'

_callback_task_hd=nil
_callback_foot_hds={}
_callback_joint_hds={}
_callback_robot_hd=nil

_callback_init_config={}

_callback_collision_hd_1 = nil
_callback_collision_hd_2 = nil

_callback_state_dim = 3

_callback_path = nil
_callback_path_index = 1
_callback_start = nil
_callback_goal = nil
_pose_generator = nil

test = 0

_matching_mode = false
_matching_pair = {}
_matching_index = 1
_init = false
_check_path={}
_applied_path={}
_is_add_to_tree = true

_pose_index = 1

_failed_time = 0

_min_dist = 0.5

type = 1
g_index = 0
sampled_states={}

_dummy_list={}

sample_from_collection = function()
    -- forbidThreadSwitches(true)
    local state = {}
    local path_state = {}
    -- if type == 1 then 
    local dim = 3
    local path_length = #_callback_path/dim

    if not _init then 
        local free_base_hd = simGetObjectHandle('free_base')
        for i=0, path_length-1, 1 do
            local ind =  i * dim
            local pos = {}
            pos[1] = _callback_path[ind + 1]
            pos[2] = _callback_path[ind + 2]
            pos[3] = 0
            simSetObjectPosition(free_base_hd, -1, pos)
            local res=simCheckCollision(free_base_hd, _callback_collision_hd_2)
            if res == 0 then
                _check_path[i] = 1
            else
                _check_path[i] = 0
            end
            _applied_path[i] = 0
        end
        _init = true
    end

    local distance = {}
    distance[1] = 0.03 + (_pose_index/path_length) * 0.05 + _failed_time * 0.05
    if distance[1] > 0.2 then 
        distance[1] = 0.2
    end    
    distance[2] = distance[1]
    distance[3] = 0.01 + (_pose_index/path_length) * 0.1 + _failed_time * 0.01
    if distance[3] > 0.5 then 
        distance[3] = 0.5
    end 

    local pose_index = _pose_index%path_length
    local index =  pose_index * dim
    path_state[1] = _callback_path[index + 1]
    path_state[2] = _callback_path[index + 2]
    path_state[3] = _callback_path[index + 3]
    if dim == 4 then
        path_state[4] = _callback_path[index + 4]
    else
        path_state[4] = 0
    end

    -- sample the faled state
    if _matching_mode then
        if (_failed_time > 1000) then 
            _pose_index = _pose_index +1
            _failed_time = 0
            _matching_pair = {}
        else
            _pose_index = _pose_index -1
            local pos1 = {}
            local pos2 = {}
            local pos3 = {}
            local pos4 = {}

            if #_matching_pair == 0 then 
                local index1 = (pose_index-3) * dim
                local index2 = (pose_index-1) * dim
                pos1[1] = _callback_path[index1 + 1]
                pos1[2] = _callback_path[index1 + 2]
                pos1[3] = _callback_path[index1 + 3]
                pos1[4] = path_state[4]

                pos2[1] = _callback_path[index2 + 1]
                pos2[2] = _callback_path[index2 + 2]
                pos2[3] = _callback_path[index2 + 3]
                pos2[4] = path_state[4]

                pos3[1] = (pos1[1] + pos2[1])/3
                pos3[2] = (pos1[2] + pos2[2])/3
                pos3[3] = (pos1[3] + pos2[1])/3
                pos3[4] = path_state[4]

                pos4[1] = (pos1[1] + pos2[1])*2/3
                pos4[2] = (pos1[2] + pos2[2])*2/3
                pos4[3] = (pos1[3] + pos2[3])*2/3
                pos4[4] = path_state[4]

                _matching_pair[1] = pos1
                _matching_pair[2] = pos3
                _matching_pair[3] = pos4
                _matching_pair[4] = pos2
                print('init!!!!')
            end
            if _failed_time%4 == 1 then 
                _matching_index = 1
            -- elseif _failed_time%4 == 2 then
            --     _matching_index = 2  
            -- elseif _failed_time%4 == 2 then
            --     _matching_index = 3  
            -- else 
            --     _matching_index = 4
            -- end
            else 
                _matching_index = math.random(4)
            end
            path_state = _matching_pair[_matching_index]
            -- path_state[4] = path_state[4] + _failed_time * 0.025  - 0.5

            _failed_time = _failed_time + 1
        end
    else
        _applied_path[pose_index -1] = 1
    end
    print('failed time: ', _applied_path[pose_index], _failed_time, _pose_index, tostring(_matching_mode))

    --------------------------------------------------

    if _check_path[pose_index] == 1 and not _matching_mode then 
        state = _callback_start
        state[1] = path_state[1]
        state[2] = path_state[2]
        _pose_index = _pose_index+1
        local r = simExtOMPL_writeState(_callback_task_hd, state)
        -- sleep(1)
        return state
    end


    local pose_collection_size = #_pose_generator.pose_list
    local candidate_pose = get_candidate_states(_pose_generator.pose_list, path_state, distance)

    print(distance[1], distance[2], distance[3])
    if #candidate_pose == 0 then
        _pose_index = _pose_index+1
        return state
    end

    local pose_collection_index = math.random(#candidate_pose)
    -- state = get_state (pose_collection_index, path_state, candidate_pose, distance) 
    state = candidate_pose[pose_collection_index]   
    local r = simExtOMPL_writeState(_callback_task_hd, state)
    -- print('found sample: '..state[1]..'  '..state[2])

    -- sleep(3)
    -- simSwitchThread()
    _pose_index = _pose_index+1
    -- forbidThreadSwitches(false)

    _is_add_to_tree = false
    return state
end

function is_free_area(pos)
    local is_free = false
    local free_base_hd = simGetObjectHandle('free_base')

    simSetObjectPosition(free_base_hd, -1, pos)
    local res=simCheckCollision(free_base_hd, _callback_collision_hd_2)
    if res == 0 then
        is_free = true
    end

    return is_free
end

get_index_need_to_sample = function(index, applied_list, check_list)
    if check_list[index] == 0 then
        return index
    end

    for i=index, #applied_list, 1 do
        if applied_list[i] == 0 then
            return i
        end
    end
    return index
end

get_state = function(path_state, state, distance)
    -- state[1] = torch.normal(path_state[1], distance[1])
    -- state[2] = torch.normal(path_state[2], distance[2])   
    -- -- state[3] = torch.normal(path_state[3], distance)    
    -- state[6] = path_state[4] --torch.normal(path_state[4], distance[3])

    local rand_x = math.random()
    local rand_y = math.random()
    local rand_yaw = math.random()
    state[1] = path_state[1] + (rand_x-0.5)*distance[1]*2
    state[2] = path_state[2] + (rand_y-0.5)*distance[2]*2
    state[6] = path_state[4] + (rand_yaw-0.5)*distance[3]*2

    return state
end

get_candidate_states = function(pose_list, path_state, distance)
    -- forbidThreadSwitches(true)

    local candidate_pose={}
    for i=1, #pose_list, 1 do
        state = get_state (path_state, pose_list[i], distance)    

        local diff_z = pose_list[i][3] - state[3]
        if diff_z > -0.1 and diff_z < 0.2 then
            -- candidate_pose[#candidate_pose+1] = pose_list[i]
            local r = simExtOMPL_writeState(_callback_task_hd, state)
            local res=simCheckCollision(_callback_collision_hd_1,_callback_collision_hd_2)
            if res == 0 then 
                candidate_pose[#candidate_pose+1] = pose_list[i]
                if #candidate_pose > 5 then
                    return candidate_pose
                end
            end
        end
    end
    -- print('candidate pose num: ', #candidate_pose, #pose_list)
    -- forbidThreadSwitches(false)

    return candidate_pose
end

sample_callback = function()
    --forbidThreadSwitches(true)
    local state = {}
    -- if type == 1 then 
    local dim = 4
    local path_length = #_callback_path/dim
    -- local pose_index = math.random(0,path_length-1)
    local pose_index = _pose_index%path_length
    local dice = math.random()

    local index =  pose_index * dim
    -- displayInfo('in callback 1 '..pose_index)
    local distance = 0.05
    -- if dice > 0.5 and #sampled_states > 2 then 
    --     -- displayInfo('pose_index2 '..#sampled_states)

    --     local pose_index2 = math.random(#sampled_states)
    --     state = sampled_states[pose_index2]
    -- else
    state[1] = _callback_path[index + 1]
    state[2] = _callback_path[index + 2]
    state[3] = _callback_path[index + 3]
    state[4] = 0
    state[5] = 0
    state[6] = 0
    state[7] = 1
    for i = 1, #_callback_init_config, 1 do 
        state[7+i] = _callback_init_config[i]
    end

    local found_pose, sampled_state = sample_state(_callback_robot_hd, _callback_joint_hds, state, distance)

    -- sleep(3)
    --simSwitchThread()

    _pose_index = _pose_index+1
    return sampled_state
end

sampleNear_callback = function(state, distance)
    -- test = 1
    -- displayInfo('in sample near!!!!!!!!!!!!!!!! '..distance)

     -- state[1] = torch.normal(path_state[1], distance[1])
    -- state[2] = torch.normal(path_state[2], distance[2])   
    -- -- state[3] = torch.normal(path_state[3], distance)    
    -- state[6] = torch.normal(path_state[4], distance[3])

    -- local rand_x = math.random()
    -- local rand_y = math.random()
    -- state[1] = state[1] + (rand_x-0.5)*distance*2
    -- state[2] = state[2] + (rand_y-0.5)*distance*2
    -- state[6] = state[6] + (rand_y-0.5)*distance*4
    -- print(#distance)
    -- local found_pose, sampled_state = sample_state(_callback_robot_hd, _callback_joint_hds, state, distance)
        
    return state
end

get_nearest_index = function(list, index)
    for i = index, #list-1, 1 do
        if list[i] == 0 then
            print(i..' '..index)
            return i
        end
    end

    for i = index, 1, -1 do
        if list[i] == 0 then
            print(i..' '..index)
            return i
        end
    end
    return index
end

sample_state=function(robot_hd, joint_hds, start_state, distance)
    local pan_hds = get_leg_pan_hds()
    -- local ikgroup_hds = get_ik_handles()

    local sample_pose = {}
    sample_pose[1] = torch.normal(start_state[1], distance)
    sample_pose[2] = torch.normal(start_state[2], distance)    
    sample_pose[3] = start_state[3] --torch.normal(start_state[3]+0.0, 0.06)    

    local sample_ori = {}
    sample_ori[1] = start_state[4] --torch.normal(start_state[4], 0.04)
    sample_ori[2] = start_state[5] --torch.normal(start_state[5], 0.04)    
    sample_ori[3] = start_state[6] --torch.normal(start_state[6], 0.05)
    sample_ori[4] = start_state[7]

    set_robot_body(robot_hd, sample_pose, sample_ori)

    marker_knee = simGetObjectHandle('temp_knee')
    marker_foot = simGetObjectHandle('temp_foot')

    local ang1s={}
    local ang2s={}
    local leg_hds={}
    local found_pose = 1
    local res = 0
    for i=1,4,1 do
        leg_hds[i]=get_leg_hds(i)
        res, ang1s[i], ang2s[i] = sample_leg_pos(i, pan_hds[i], leg_hds[i])
        found_pose = found_pose*res
    end
    
    if found_pose == 0 then
        -- sampled_state = start_state
        -- set_robot_body(robot_hd, startpos, startorient)
        -- set_joint_positions(joint_hds, startconfigs)
        local r = simExtOMPL_writeState(_callback_task_hd, start_state)
    end

    local res, sampled_state = simExtOMPL_readState(_callback_task_hd)
    --displayInfo('in sample_state '..#sampled_state)

    -- sleep(2)
    -- simSwitchThread()
    return found_pose, sampled_state
end

sample_leg_pos=function(index, pan_hd, leg_hds)
    local pan_pos = simGetJointPosition(pan_hd, -1)
    -- local sample_pan = torch.normal(pan_pos, 1.575)
    -- local sample_r = torch.normal(0.0, 0.05)

    local sample_r = math.random(-0.08, 0.08)
    local sample_pan = math.random(pan_pos-1.575, pan_pos+1.575)

    simSetJointPosition(pan_hd, sample_pan)

    local pos = simGetObjectPosition(leg_hds[1], -1)
    local knee_pos = simGetObjectPosition(leg_hds[2], leg_hds[1])

    local r0 = 0.07238 --math.sqrt(knee_pos[1]^2 + knee_pos[2]^2)
    local r1 = 0.10545 --math.sqrt(tip_pos[1]^2 + tip_pos[2]^2)

    local tip_pos={}
    tip_pos[1] = sample_r
    tip_pos[2] = pos[3] - 0.0475
    tip_pos[3] = 0
    simSetObjectPosition(marker_foot, leg_hds[1], tip_pos)

    local knee_x, knee_y = get_intersection_point(0, 0, tip_pos[1], tip_pos[2], r0, r1)
    if knee_x == -1 then
        return 0, 0, 0
    end
    local knee_pos={}
    knee_pos[1] = knee_x
    knee_pos[2] = knee_y
    knee_pos[3] = 0
    simSetObjectPosition(marker_knee, leg_hds[1], knee_pos)

    local tip_x_fromknee = tip_pos[1]-knee_x
    local tip_y_fromknee = tip_pos[2]-knee_y

    --displayInfo('knee pos: '..tip_x_fromknee..' '..tip_y_fromknee)


    local angle_thigh = math.atan(knee_y/knee_x)
    local angle_knee = math.atan(tip_y_fromknee/tip_x_fromknee)
    if angle_knee<0 then 
        angle_knee = angle_knee + math.pi 
    end

    simSetJointPosition(leg_hds[1], angle_thigh)
    simSetJointPosition(leg_hds[2], angle_knee-angle_thigh)

    local pose_real = simGetObjectPosition(leg_hds[3], leg_hds[1])

    local error_x = math.abs(tip_pos[1]-pose_real[1])
    local error_y = math.abs(tip_pos[2]-pose_real[2])
    local error_z = math.abs(tip_pos[3]-pose_real[3])

    local good_pos = 1
    if error_x > 0.01 or error_y > 0.01 or error_z > 0.01 then
        good_pos = 0
    end
    --displayInfo('good pos: '..good_pos)

    -- simSwitchThread()


    -- local target_matrix=simGetObjectMatrix(marker_foot,-1)
    -- local jointPositions = check_ik(target_matrix, tip_hd, ik_group_hd, leg_hds)

    -- if jointPositions == nil then
    --     --displayInfo('not found: ')
    -- else
    --     set_joint_positions(leg_hds, jointPositions)
    -- end
    return good_pos, angle_thigh, angle_knee-angle_thigh
end
 
stateValidation=function(state)
    -- displayInfo('in stateValidation ')

    -- Read the current state:
    --local res, current_state = simExtOMPL_readState(_task_hd)
    --_sample_num = _sample_num+1
    local r = simExtOMPL_writeState(_callback_task_hd, state)
    local pass=false
    
    -- check if the foot is on the ground
    local isOnGround = true
    -- foot_pos = get_foottip_positions(_callback_foot_hds)
    -- for i=1,#foot_pos,1 do
    --     local pos = foot_pos[i]
    --     if pos[3] > 0.03 then
    --         isOnGround = false
    --         break
    --     end
    -- end

    if isOnGround then
        local res=simCheckCollision(_callback_collision_hd_1,_callback_collision_hd_2)
        --local res, dist = simCheckDistance(simGetCollectionHandle('robot_body'),simGetCollectionHandle('obstacles'),0.02)
        if res == 0 then
            pass=true
            --_valid_num = _valid_num+1
        end
    end
    --res = simExtOMPL_writeState(_task_hd, current_state)
    -- sleep(1)
    -- simSwitchThread()
    --displayInfo('callback: '..test)

    -- Return whether the tested state is valid or not:
    -- print('stateValidation: '..state[1], state[2], tostring(pass))
    -- _is_add_to_tree = pass

    return pass
end

quick_motionValidation = function(state_tree, state, valid)
    local check_motion = true

    local diff_x = math.abs(state_tree[1] - state[1])
    local diff_y = math.abs(state_tree[2] - state[2])

    -- if diff_x > 0.3 or diff_y > 0.3 then 
    --     check_motion = false
    -- end

    -- print('qmc '..state_tree[1]..'  '..state_tree[2]..'  '..state[1]..'  '..state[2]..'  '..tostring(check_motion))
    -- _is_add_to_tree = tru
    return check_motion
end

motionValidation=function(state_tree, state, valid)
    
    local diff_x = math.abs(state_tree[1] - state[1])
    local diff_y = math.abs(state_tree[2] - state[2])

    local dist = math.sqrt(diff_x*diff_x + diff_y*diff_y)

    if valid then 
        if _matching_index ~= 1 then 
            _matching_mode = false
            _matching_pair = {}
        end
        _is_add_to_tree = true

        if not _matching_mode then
            _failed_time = 0
        end
        -- local hd1 = simGetObjectHandle('state_tree')
        -- local hd2 = simGetObjectHandle('state')

        local pos1={}
        local pos2={}
        local ori={}

        for i=1, 3, 1 do
            pos1[i] = state_tree[i]
            pos2[i] = state[i]
        end
        for i=4, 7, 1 do
            ori[i-3] = state[i]
        end        

        -- simSetObjectPosition(hd1, -1, pos2)
        -- simSetObjectPosition(hd2, -1, pos2)
        -- sleep(1)
        -- simSwitchThread()
        -- create_dummy(pos2, ori)
        print ('motion validation: '.._matching_index..' '..tostring(valid))


    else        
        if dist < _min_dist then 
            _matching_mode = true
        end  
    end
    -- print ('motion validation: '.._matching_index..' '..tostring(valid))

    return true
end


goalSatisfied = function(state)
    local satisfied=0
    local dist=0
    local diff={}
    for i=1, #_callback_goal, 1 do
        diff[i]=math.abs(state[i]-_callback_goal[i])
    end

    local min_dist = 0.1
    if diff[1] < min_dist and diff[2] < min_dist then
    -- if state[1]-_callback_goal[1] < 0.05 and state[2]-_callback_goal[2] < 0.1 then
        satisfied=1
    end

    dist=diff[3]+diff[4]+diff[5]+diff[6]+diff[7]
    return satisfied, dist
end


check_ik=function(target_matrix, tip_hd, ik_group_hd, jh)
    simSetObjectMatrix(tip_hd,-1,target_matrix)
    local jointPositions = simGetConfigForTipPose(ik_group_hd,jh,0.05,10)
    return jointPositions
end


get_intersection_point=function(x0, y0, x1, y1, r0, r1)
    local d=math.sqrt((x1-x0)^2 + (y1-y0)^2)
    if d>(r0+r1) then
        return -1, -1
    end
    
    local a=(r0^2-r1^2+d^2)/(2*d)
    local h=math.sqrt(r0^2-a^2)
    local x2=x0+a*(x1-x0)/d   
    local y2=y0+a*(y1-y0)/d   
    local x3=x2+h*(y1-y0)/d       -- also x3=x2-h*(y1-y0)/d
    local y3=y2-h*(x1-x0)/d       -- also y3=y2+h*(x1-x0)/d

    return x3, y3
end

function create_dummy(pos, ori)
    local hd = simCreateDummy(0.1)
    pos[3] = pos[3]+0.4
    simSetObjectPosition(hd, -1, pos)
    simSetObjectQuaternion(hd, -1, ori)
    _dummy_list[#_dummy_list+1] = hd
end