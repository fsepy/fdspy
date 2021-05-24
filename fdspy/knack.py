import knackpy
import requests
import json
import time
from typing import List


class KnackFDSpy:
    def __init__(
            self,
            app_id: str,
            api_key: str,
            fetch_view_id: str,
            put_object_id: str,
            tzinfo='Europe/London',
            timeout:int=10,
    ):
        self.__app_private = knackpy.App(
            app_id = app_id,
            api_key = api_key,
            tzinfo=tzinfo,
            timeout=timeout
        )
        self.__app_public = knackpy.App(
            app_id=app_id,
            api_key='knack',
            tzinfo=tzinfo,
            timeout=timeout
        )
        self.__fetch_view_id = fetch_view_id
        self.__put_object_id = put_object_id

        self.__headers = {
            "X-Knack-Application-Id": app_id,
            "X-Knack-REST-API-Key": api_key,
            "Content-Type": "application/json",
        }

    def get_records(self) -> List[dict]:
        records = self.__app_public.get(self.__fetch_view_id, refresh=True)
        return [dict(i) for i in records]

    def put_record(self, obj: str, id: str, data: dict):

        rp = requests.put(
            url=f'https://api.knack.com/v1/objects/{obj}/records/{id}',
            data=json.dumps(data),
            headers=self.__headers,
        )

        return rp

    def run(self, sleep: float = 1):
        for i in range(100):

            records = self.get_records()

            print(i)

            for record in records:
                print(i, record['field_590'])
                if record['field_590'] == 'Pending':
                    print(i, f'{record["id"]} is Pending, updated to Submitted.')
                    self.put_record(obj=self.__put_object_id, id=record['id'], data=dict(field_590='Submitted'))

            time.sleep(sleep)


if __name__ == '__main__':

    app = KnackFDSpy(
        app_id="59c2dafe9934032bf00cb699",
        api_key="61f123b0-9f9b-11e7-b36d-172175cd1e5a",
        fetch_view_id='view_837',
        put_object_id='object_56'
    )

    app.run()

    # app = knackpy.App(
    #     app_id="59c2dafe9934032bf00cb699",
    #     api_key="61f123b0-9f9b-11e7-b36d-172175cd1e5a",
    #     tzinfo='Europe/London',
    #     timeout=10
    # )
    # # records = app.get("view_837")
    # records = app.get("object_56")
    # for record in records:
    #     print(record.format())
    #
    # data = dict(records[0])
    # data['field_587'] = 60
    #
    # print(data)
    #
    # # record = app.record(method="update", data=data, obj="object_56")
    #
    # headers = {
    #     "X-Knack-Application-Id": "59c2dafe9934032bf00cb699",
    #     "X-Knack-REST-API-Key": "61f123b0-9f9b-11e7-b36d-172175cd1e5a",
    #     "Content-Type": "application/json",
    # }
    #
    # rp = requests.put(
    #     url=f'https://api.knack.com/v1/objects/object_56/records/{data["id"]}',
    #     data=json.dumps({'field_587': 60, 'field_590': 'Running'}),
    #     headers=headers,
    # )
    #
    # rp.content