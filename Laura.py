def read_CPC(path, parent_path, timelable):
     parentPath = os.path.abspath(parent_path)
     if parentPath not in sys.path:
        sys.path.insert(0, parentPath)
        
        