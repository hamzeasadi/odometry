"""
pytest for all possible solutions
"""

from Engine.motion import MotionQueue




def test_motion_queue():
    """
    test the custome queue
    
    """
    mq = MotionQueue(mq_size=10)
    for i in range(12):
        mq.push((i+1))

    assert mq.pop() == 3
