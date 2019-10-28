app_id = r"59c2dafe9934032bf00cb699"
api_key = r"61f123b0-9f9b-11e7-b36d-172175cd1e5a"


if __name__ == "__main__":
    from knackpy import Knack

    kn = Knack(
        # scene='object_37',
        scene="174",
        view="view_183",
        app_id=app_id,
        api_key=api_key,
        # page_limit=2,
        # rows_per_page=10
    )

    print(kn)
