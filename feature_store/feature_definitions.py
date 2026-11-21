
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.value_type import ValueType

# Each traffic sensor is identified by its sensor ID.
sensor = Entity(
    name="sensor",
    join_keys=["sensor_id"],
    value_type=ValueType.INT64,
    description="Roadside traffic sensor"
)


traffic_source = FileSource(
    name="traffic_data_source",
    path="data/traffic_features.parquet",
    timestamp_field="event_timestamp"
)


# Traffic features available for training and prediction.
traffic_features = FeatureView(
    name="traffic_features",
    entities=[sensor],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="flow", dtype=Float32),
        Field(name="occupancy", dtype=Float32),
        Field(name="speed", dtype=Float32),
    ],
    source=traffic_source,
    online=True
)
