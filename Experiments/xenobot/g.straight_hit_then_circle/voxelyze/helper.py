def largest_component(body):
    """return the largest component of the body (get rid of those unconnected parts)"""
    from scipy.ndimage.measurements import label
    import numpy as np
    labeled, total = label(body)
    largest_component_id = 0
    largest_component_count = 0
    if total>10000:
        print(f"Warning: {__file__} largest_component() is not efficient with too many components.")
    for i in range(1,total+1):
        count = np.sum(labeled==i)
        if count>largest_component_count:
            largest_component_count = count
            largest_component_id = i
    ret = np.zeros_like(body)
    ret[labeled==largest_component_id] = body[labeled==largest_component_id]
    return ret

def mkdir_if_not_exist(dir):
    import os
    if not os.path.exists(dir):
        os.mkdir(dir)

def run_shell_command(commandline):
    import os
    os.system(commandline)

def send_slack(msg):
    """This is a private function, you can make one of your own, so the program will send message to our slack channel during training."""
    try:
        import sida.slackbot.bot as bot
        bot.send(msg,1,"GUB0XS56E") # send msg to #gpuvoxel_gpu_1
    except:
        pass
    
if __name__ == "__main__":
    # testing
    def test_largest_component():
        import numpy as np
        body = np.zeros([3,3,3], dtype=int)
        body[0,1,1:] = 1
        body[0,0,0] = 1
        body[2,2,2] = 1
        print(body)
        body = largest_component(body)
        print(body)
    #test_largest_component()
    def test_mkdir_if_not_exist():
        import os
        import shutil
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        mkdir_if_not_exist("tmp")
        mkdir_if_not_exist("tmp/tmp")
        shutil.rmtree("tmp")
    # test_mkdir_if_not_exist()
    def test_run_shell_command():
        mkdir_if_not_exist("tmp")
        run_shell_command("echo hello,shell > tmp/hello")
    # test_run_shell_command()
    def test_send_slack():
        send_slack("test")
    test_send_slack()