if __name__ == '__main__':
    from fsetoolsGUI.fds.read_sf.test_files import fp_fds

    from fsetoolsGUI.f90nml import read

    namelist = read(fp_fds)
    print(namelist['head']['chid'])
    print(len(namelist['head']))

    from fsetoolsGUI.f90nml.namelist import Namelist

    a = Namelist()
    a['hello'] = Namelist()
    a['hello'] = Namelist()
    print(a)
