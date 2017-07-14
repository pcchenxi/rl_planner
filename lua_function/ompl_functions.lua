package.path=package.path .. ";/home/xi/workspace/v-rep_code/rl_planner/lua_functions/?.lua"

require("ompl_callbacks")
require("common_functions")

--------------------------------------------- ompl functions -------------------------------------------------------
_robot_dim = 2
_joint_dim = 4
_collision_hd_1 = nil
_collision_hd_2 = nil
_use_validation_callback = false
_use_sampler_callback = false
_use_collision_check = true
_callback_nume = 'stateValidation'

_path = {}

_state_dim = 0

_sample_num     = 0
_valid_num      = 0

init_statespace=function(robot_hd, joint_hds, start_pose, goal_pose)
    local state_spaces={}
    local min_v = 0
    local max_v = 0
    local weight = 0
    
    local min_y = -2 --start_pose[2]-1.5
    local max_y = 2  --goal_pose[2]+1.5

    local min_x = -2
    local max_x = 2

    local min_z = 0
    local max_z = 0

    if _robot_dim > 1 then
        min_z = start_pose[3] - 0.1
        max_z = start_pose[3] + 0.1
    end
    
    if _robot_dim == 1 then
        min_range={min_x,min_y}
        max_range={max_x,max_y}
        local weight_move = 3
        state_spaces[1]=simExtOMPL_createStateSpace('base_space',sim_ompl_statespacetype_position2d,robot_hd,min_range,max_range,weight_move)               -- base space
    elseif _robot_dim == 2 then
        min_range={min_x,min_y}
        max_range={max_x,max_y}
        local weight_move = 3
        state_spaces[1]=simExtOMPL_createStateSpace('base_space',sim_ompl_statespacetype_pose2d,robot_hd,min_range,max_range,weight_move)               -- base space
    elseif _robot_dim == 3 then
        min_range={min_x,min_y, min_z}
        max_range={max_x,max_y, max_z}
        local weight_move = 3
        state_spaces[1]=simExtOMPL_createStateSpace('base_space',sim_ompl_statespacetype_position3d,robot_hd,min_range,max_range,weight_move)               -- base space
    elseif _robot_dim == 4 then
        min_range={min_x,min_y, min_z}
        max_range={max_x,max_y, max_z}
        local weight_move = 3
        state_spaces[1]=simExtOMPL_createStateSpace('base_space',sim_ompl_statespacetype_position3d,robot_hd,min_range,max_range,weight_move)               -- base space

        local yaw_hd = simGetObjectHandle('base_yaw')        
        min_v = {0*math.pi/180}
        max_v = {90*math.pi/180}
        state_spaces[2]=simExtOMPL_createStateSpace('base_yaw',sim_ompl_statespacetype_joint_position,yaw_hd,min_v,max_v,weight,robot_hd)

    elseif _robot_dim == 6 then
        min_range={min_x,min_y, min_z}
        max_range={max_x,max_y, max_z}
        local weight_move = 3
        state_spaces[1]=simExtOMPL_createStateSpace('base_space',sim_ompl_statespacetype_pose3d,robot_hd,min_range,max_range,weight_move)               -- base space        
    end

    for i=1,_joint_dim,1 do
        if i < 5 then
            min_v = {-90*math.pi/180}    --{-170*math.pi/180}
            max_v = {90*math.pi/180}    --{170*math.pi/180}
            weight = 0
        elseif i < 9 then
            min_v = {-90*math.pi/180}    --{-170*math.pi/180}
            max_v = {90*math.pi/180}    --{170*math.pi/180}
            weight = 0
        else 
            min_v = {-90*math.pi/180}    --{-170*math.pi/180}
            max_v = {90*math.pi/180}    --{170*math.pi/180}
            weight = 0
        end
        state_spaces[#state_spaces+1]=simExtOMPL_createStateSpace('joint'..i,sim_ompl_statespacetype_joint_position,joint_hds[i],min_v,max_v,weight,robot_hd)
    end

    return state_spaces
end


init_params=function(robot_dim, joint_dim, collision_name1, collision_name2, use_validation_callback, use_sampler_callback)
    set_robot_dim(robot_dim)
    set_joint_dim(joint_dim)
    set_collision_hd(collision_name1, collision_name2)
    set_use_callback(use_validation_callback)
    _use_sampler_callback = use_sampler_callback
end

init_task=function(start_name, task_id)
    local robot_hd       =simGetObjectHandle(start_name)
    local target_hd      =simGetObjectHandle('target')
    local joint_hds      =get_joint_hds()

    local task_hd = simExtOMPL_createTask(task_id)
    -- simExtOMPL_setVerboseLevel(task_hd, 3)
    simExtOMPL_setAlgorithm(task_hd,sim_ompl_algorithm_RRTConnect)

    ------ callbacks ---------------\
    simExtOMPL_setGoalCallback(task_hd, 'goalSatisfied')
    _callback_task_hd = task_hd
    _callback_collision_hd_1 = _collision_hd_1
    _callback_collision_hd_2 = _collision_hd_2    
    if _use_validation_callback then
        _callback_foot_hds = get_foot_tip_hds()
        simExtOMPL_setStateValidationCallback(task_hd, 'stateValidation')
    end

    if _use_sampler_callback then 
        simExtOMPL_setMotionValidationCallback(task_hd, 'motionValidation', 'quick_motionValidation')
        _callback_joint_hds = joint_hds
        _callback_robot_hd = robot_hd
        _callback_path = _path
        _callback_state_dim = _state_dim
        _callback_init_config = get_joint_positions(joint_hds, _joint_dim)

        pose_gen = Pose_Generator:new()
        pose_gen:init_pose_list()
        _pose_generator = pose_gen
        local r = simExtOMPL_setValidStateSamplerCallback(task_hd, 'sample_from_collection', 'sampleNear_callback')       

        -- local r = simExtOMPL_setValidStateSamplerCallback(task_hd, 'sample_callback', 'sampleNear_callback')       

        -- simExtOMPL_setAlgorithm(task_hd,sim_ompl_algorithm_RRTstar)

        -- displayInfo('use callback '..#_path)  sample_from_collection

    end

    -- start pose --
    local startpose=get_robot_pose(robot_hd, joint_hds, _robot_dim, _joint_dim)
    -- target pose --
    local goalpose=get_robot_pose(target_hd, joint_hds, _robot_dim, _joint_dim)
    _state_dim = #startpose
    _callback_start = startpose
    _callback_goal = goalpose

    satat_spaces=init_statespace(robot_hd, joint_hds, startpose, goalpose) -- for sample state
    simExtOMPL_setStateSpace(task_hd, satat_spaces)

    simExtOMPL_setCollisionPairs(task_hd, {_collision_hd_1, _collision_hd_2}) -- collision 

    simExtOMPL_setStartState(task_hd, startpose)    
    simExtOMPL_setGoalState(task_hd, goalpose)

    local txt='state_space dim ' ..#startpose
    -- displayInfo(txt)
    -- simExtOMPL_printTaskInfo(task_hd)
    return task_hd, _state_dim
end

compute_path=function(task_hd, max_time)
    -- forbidThreadSwitches(true)
    r,_path=simExtOMPL_compute(task_hd, max_time, -1, 50)
    -- forbidThreadSwitches(false)

    path_step = #_path/_state_dim
    --local txt='finish compute' ..path_step
    --displayInfo(txt)
    
    return _path
end

set_robot_dim=function(robot_dim)
    _robot_dim = robot_dim
end

set_joint_dim=function(joint_dim)
    _joint_dim = joint_dim
end

set_collision_hd=function(name1, name2)
    _collision_hd_1 = simGetCollectionHandle(name1)
    _collision_hd_2 = simGetCollectionHandle(name2)
end

set_use_callback=function(use_validation_callback, function_name)
    _use_validation_callback = use_validation_callback
    _callback_nume = function_name
end



