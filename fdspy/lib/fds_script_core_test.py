from fds_script_core import *


class Test_GeoOrtho(GeoOrtho):
    cls = GeoOrtho()

    def test_2d(self):
        def check(ans, *args, **kwargs):
            res = self.cls.state2d(*args, **kwargs)
            print(res, ans)
            assert res == ans

        # Cases no contact
        ans = 0
        (x1, y1), (x2, y2) = (0, 0), (1, 1)
        (x3, y3), (x4, y4) = (2, 0), (3, 1)
        check(ans=ans, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4)

        (x1, y1), (x2, y2) = (0, 0), (1, 1)
        (x3, y3), (x4, y4) = (2, 2), (3, 3)
        check(ans=ans, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4)

        # Cases contact on single point
        ans = 1
        (x1, y1), (x2, y2) = (0, 0), (1, 1)
        (x3, y3), (x4, y4) = (1, 0), (2, -1)
        check(ans=ans, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4)

        (x1, y1), (x2, y2) = (0, 0), (1, 1)
        (x3, y3), (x4, y4) = (1, 1), (2, 2)
        check(ans=ans, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4)

        # Cases contact on edge
        ans = 2
        (x1, y1), (x2, y2) = (0, 0), (1, 1)
        (x3, y3), (x4, y4) = (1, 0), (2, 1)
        check(ans=ans, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4)

        (x1, y1), (x2, y2) = (0, 0), (1, 1)
        (x3, y3), (x4, y4) = (1, -0.5), (3, 1.5)
        check(ans=ans, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4)

        (x1, y1), (x2, y2) = (0, 0), (1, 1)
        (x3, y3), (x4, y4) = (1, 0.5), (2, 1.5)
        check(ans=ans, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4)

        # Cases overlap
        ans = 3
        (x1, y1), (x2, y2) = (0, 0), (1, 1)
        (x3, y3), (x4, y4) = (0.5, 0.5), (1.5, 1.5)
        check(ans=ans, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4)

        (x1, y1), (x2, y2) = (0, 0), (1, 1)
        (x3, y3), (x4, y4) = (0.5, 0), (1.5, 1)
        check(ans=ans, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4)

        (x1, y1), (x2, y2) = (0, 0), (1, 1)
        (x3, y3), (x4, y4) = (0, 0), (1, 1)
        check(ans=ans, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4)

        (x1, y1), (x2, y2) = (0, 0), (1, 1)
        (x3, y3), (x4, y4) = (0.25, 0.25), (0.75, 0.75)
        check(ans=ans, x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, x4=x4, y4=y4)

    def test_3d(self):
        def check(ans, *args, **kwargs):
            res = self.state3d(*args, **kwargs)
            print(res, ans)
            assert res == ans

        # =====
        ans = 0  # Check for no contact
        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (2, 0, 0), (3, 1, 1)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (2, 2, 2), (3, 3, 3)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        # =====
        ans = 1  # Check for point contact
        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (1, 1, 1), (2, 2, 2)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (-1, -1, -1), (0, 0, 0)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        # =====
        ans = 2  # Check for edge contact
        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (1, 1, 0), (2, 2, 1)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (1, 1, 0.5), (2, 2, 1.5)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        # =====
        ans = 3  # Check for face contact
        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (1, 0, 0), (2, 1, 1)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (1, -0.5, 0), (3, 1.5, 1)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (1, 0.5, 0), (2, 1.5, 1)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        # =====
        ans = 4  # Check for face contact
        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (0.5, 0.5, 0.5), (1.5, 1, 1)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (0.5, 0, 0), (1.5, 1, 1)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (0, 0, 0), (1, 1, 1)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (0.25, 0.25, 0), (0.75, 0.75, 1)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)

        (x1, y1, z1), (x2, y2, z2) = (0, 0, 0), (1, 1, 1)
        (x3, y3, z3), (x4, y4, z4) = (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)
        check(ans, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)


class Test_MESH:
    def test_1(self):
        pass


class Test_FDSBaseModel:
    cls = FDSBaseModel()

    def test_fds2list_single_line(self):
        fds2list_single_line = self.cls._fds2list_single_line

        def len_fds2list_single_line(line_):
            line_ = fds2list_single_line(line_)
            if isinstance(line_, list):
                return len(line_)
            elif line_ is None:
                return None

        line = r"&HEAD CHID='moe1'/"
        print(fds2list_single_line(line))
        assert len_fds2list_single_line(line) == 3

        line = r"&TIME T_END=400.0, RESTRICT_TIME_STEP=.FALSE./"
        print(fds2list_single_line(line))
        assert len_fds2list_single_line(line) == 5

        line = r"&MESH ID='stair upper02', IJK=7,15,82, XB=4.2,4.9,-22.0,-20.5,11.1,19.3, MPI_PROCESS=0/"
        print(fds2list_single_line(line))
        assert len_fds2list_single_line(line) == 9

        line = r"""
        &PART ID='Tracer',
              MASSLESS=.TRUE.,
              MONODISPERSE=.TRUE.,
              AGE=60.0/
        """
        print(fds2list_single_line(line))
        assert len_fds2list_single_line(line) == 9

        line = r"&CTRL ID='invert', FUNCTION_TYPE='ALL', LATCH=.FALSE., INITIAL_STATE=.TRUE., INPUT_ID='ventilation'/"
        print(fds2list_single_line(line))
        assert len_fds2list_single_line(line) == 11

        line = r"&HOLE ID='door - stair_bottom', XB=3.0,3.4,-23.1,-22.3,4.9,6.9/ "
        print(fds2list_single_line(line))
        assert len_fds2list_single_line(line) == 5

        line = r"&SLCF QUANTITY='TEMPERATURE', VECTOR=.TRUE., PBX=3.4/"
        print(fds2list_single_line(line))
        assert len_fds2list_single_line(line) == 7

        line = r"&TAIL /"
        print(fds2list_single_line(line))
        assert len_fds2list_single_line(line) == 1

        line = r"""
        &SURF ID='LINING CONCRETE',
              COLOR='GRAY 80',
              BACKING='VOID',
              MATL_ID(1,1)='CONCRETE',
              MATL_MASS_FRACTION(1,1)=1.0,
              THICKNESS(1)=0.2/
        """
        print(fds2list_single_line(line))
        assert len_fds2list_single_line(line) == 13

        line = r"""&TIME T_END=400.0, RESTRICT_TIME_STEP=.FALSE./"""
        print(fds2list_single_line(line))
        assert fds2list_single_line(line)[3] == "RESTRICT_TIME_STEP"

    def test_fds2df(self):
        from fdspy.tests.fds_scripts import general_residential_corridor
        self.cls.fds_raw = general_residential_corridor
        print(self)

    def test_df2fds(self):
        from fdspy.tests.fds_scripts import travelling_fire_1cw
        self.fds_raw = travelling_fire_1cw

        df_fds = self.cls.fds_df.copy()
        print(df_fds.loc[df_fds['_GROUP'] == 'MESH'])

        meshes = list()
        for index, row in df_fds.loc[df_fds['_GROUP'] == 'MESH'].iterrows():
            row.dropna(inplace=True)
            line_dict = row.to_dict()
            meshes.append(MESH(**line_dict))

        print(meshes)

        edges = list()
        for i, mesh in enumerate(meshes):
            edges.append(list())
            for j, mesh_ in enumerate(meshes):
                if mesh < mesh_:
                    edges[-1].append(j)
        print(edges)

        weights = [i.size_cell() for i in meshes]
        print(weights)


if __name__ == '__main__':
    Test_GeoOrtho()
    Test_FDSBaseModel()
