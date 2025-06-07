define lidar ranger
(
  sensor
  (
    range [0 20]
    fov 360.0
    samples 721
  )
    model
  (
    # generic model properties
    pose [ 0.0 0.0 -0.2 0.0 ]
    size [ 0.07 0.07 0.05 ]  
    color "black"
  )
)

define rosbot position
(
  #localization "gps"
  localization "odom"
  odom_error [0.0 0.01 0.0 0.01 0.0 0.01]

  block( 
    points 8
    point[0] [-0.2 0.12]
    point[1] [-0.2 -0.12]
    point[2] [-0.12 -0.2555]
    point[3] [0.12 -0.2555]
    point[4] [0.2 -0.12]
    point[5] [0.2 0.12]
    point[6] [0.12 0.2555]
    point[7] [-0.12 0.2555]
    z [0 0.22]
  )
  origin [ 0.0 0.0 0.0 0.0 ]
  
  gui_nose 1
  drive "diff"
  color "red"
  obstacle_return 1    # Can hit things.
  laser_return 1
  
  lidar(pose [ 0.0 0.0 0.0 180.0 ])
)


