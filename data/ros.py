import rospy
from std_msgs.msg import String


def door_open_callback(data):
    rospy.loginfo(rospy.get_caller_id() + " I heard %s", data.data)
    # Tell video recognition script that new people might have entered the vehicle


def door_close_callback(data):
    rospy.loginfo(rospy.get_caller_id() + " I heard %s", data.data)
    # Tell video recognition script that people might have left the vehicle


def listener():
    rospy.init_node("door_listener", anonymous=True)

    rospy.Subscriber("/can/opening_door", String, door_open_callback)
    rospy.Subscriber("/can/closing_door", String, door_close_callback)

    rospy.spin()


if __name__ == "__main__":
    listener()
