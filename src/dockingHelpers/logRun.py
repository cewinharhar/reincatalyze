def logRun(log_file_path, message):
    with open(log_file_path, "a+") as f:
        f.write(message + "\n")
    f.close()
