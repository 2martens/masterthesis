# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: scenenet.proto

import sys

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode('latin1'))

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
    name='scenenet.proto',
    package='scenenet',
    serialized_pb=_b(
        '\n\x0escenenet.proto\x12\x08scenenet\"\xa6\x01\n\x0bSceneLayout\x12\x35\n\x0blayout_type\x18\x01 \x01('
        '\x0e\x32 .scenenet.SceneLayout.LayoutType\x12\r\n\x05model\x18\x02 \x01('
        '\t\"Q\n\nLayoutType\x12\x0c\n\x08\x42\x41THROOM\x10\x01\x12\x0b\n\x07\x42\x45\x44ROOM\x10\x02\x12\x0b\n'
        '\x07KITCHEN\x10\x03\x12\x0f\n\x0bLIVING_ROOM\x10\x04\x12\n\n\x06OFFICE\x10\x05\"\x87\x02\n\tLightInfo\x12'
        '\x31\n\nlight_type\x18\x01 \x01(\x0e\x32\x1d.scenenet.LightInfo.LightType\x12%\n\x0clight_output\x18\x02 '
        '\x01(\x0b\x32\x0f.scenenet.Power\x12$\n\x08position\x18\x03 \x01('
        '\x0b\x32\x12.scenenet.Position\x12\x0e\n\x06radius\x18\x04 \x01(\x02\x12\x1e\n\x02v1\x18\x05 \x01('
        '\x0b\x32\x12.scenenet.Position\x12\x1e\n\x02v2\x18\x06 \x01('
        '\x0b\x32\x12.scenenet.Position\"*\n\tLightType\x12\n\n\x06SPHERE\x10\x01\x12\x11\n\rPARALLELOGRAM\x10\x02'
        '\"\xb0\x03\n\x10RandomObjectInfo\x12\x15\n\rshapenet_hash\x18\x01 \x01(\t\x12\x15\n\rheight_meters\x18\x02 '
        '\x01(\x02\x12>\n\x0bobject_pose\x18\x03 \x01('
        '\x0b\x32).scenenet.RandomObjectInfo.Transformation\x1a\xad\x02\n\x0eTransformation\x12\x15\n\rtranslation_x'
        '\x18\x01 \x01(\x02\x12\x15\n\rtranslation_y\x18\x02 \x01(\x02\x12\x15\n\rtranslation_z\x18\x03 \x01('
        '\x02\x12\x16\n\x0erotation_mat11\x18\x04 \x01(\x02\x12\x16\n\x0erotation_mat12\x18\x05 \x01('
        '\x02\x12\x16\n\x0erotation_mat13\x18\x06 \x01(\x02\x12\x16\n\x0erotation_mat21\x18\x07 \x01('
        '\x02\x12\x16\n\x0erotation_mat22\x18\x08 \x01(\x02\x12\x16\n\x0erotation_mat23\x18\t \x01('
        '\x02\x12\x16\n\x0erotation_mat31\x18\n \x01(\x02\x12\x16\n\x0erotation_mat32\x18\x0b \x01('
        '\x02\x12\x16\n\x0erotation_mat33\x18\x0c \x01(\x02\"\xc0\x02\n\x08Instance\x12\x13\n\x0binstance_id\x18\x01 '
        '\x01(\x05\x12\x1b\n\x13semantic_wordnet_id\x18\x02 \x01(\t\x12\x18\n\x10semantic_english\x18\x03 \x01('
        '\t\x12\x36\n\rinstance_type\x18\x04 \x01('
        '\x0e\x32\x1f.scenenet.Instance.InstanceType\x12\'\n\nlight_info\x18\x05 \x01('
        '\x0b\x32\x13.scenenet.LightInfo\x12/\n\x0bobject_info\x18\x06 \x01('
        '\x0b\x32\x1a.scenenet.RandomObjectInfo\"V\n\x0cInstanceType\x12\x0e\n\nBACKGROUND\x10\x01\x12\x11\n'
        '\rLAYOUT_OBJECT\x10\x02\x12\x10\n\x0cLIGHT_OBJECT\x10\x03\x12\x11\n\rRANDOM_OBJECT\x10\x04\"('
        '\n\x05Power\x12\t\n\x01r\x18\x01 \x01(\x02\x12\t\n\x01g\x18\x02 \x01(\x02\x12\t\n\x01\x62\x18\x03 \x01('
        '\x02\"+\n\x08Position\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 '
        '\x01(\x02\"a\n\x04Pose\x12\"\n\x06\x63\x61mera\x18\x01 \x01('
        '\x0b\x32\x12.scenenet.Position\x12\"\n\x06lookat\x18\x02 \x01('
        '\x0b\x32\x12.scenenet.Position\x12\x11\n\ttimestamp\x18\x03 \x01('
        '\x02\"f\n\x04View\x12\x11\n\tframe_num\x18\x01 \x01(\x05\x12$\n\x0cshutter_open\x18\x02 \x01('
        '\x0b\x32\x0e.scenenet.Pose\x12%\n\rshutter_close\x18\x03 \x01('
        '\x0b\x32\x0e.scenenet.Pose\"\x8e\x01\n\nTrajectory\x12%\n\x06layout\x18\x01 \x01('
        '\x0b\x32\x15.scenenet.SceneLayout\x12%\n\tinstances\x18\x02 \x03('
        '\x0b\x32\x12.scenenet.Instance\x12\x1d\n\x05views\x18\x03 \x03('
        '\x0b\x32\x0e.scenenet.View\x12\x13\n\x0brender_path\x18\x04 \x01('
        '\t\":\n\x0cTrajectories\x12*\n\x0ctrajectories\x18\x01 \x03(\x0b\x32\x14.scenenet.Trajectory')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

_SCENELAYOUT_LAYOUTTYPE = _descriptor.EnumDescriptor(
    name='LayoutType',
    full_name='scenenet.SceneLayout.LayoutType',
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name='BATHROOM', index=0, number=1,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='BEDROOM', index=1, number=2,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='KITCHEN', index=2, number=3,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='LIVING_ROOM', index=3, number=4,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='OFFICE', index=4, number=5,
            options=None,
            type=None),
    ],
    containing_type=None,
    options=None,
    serialized_start=114,
    serialized_end=195,
)
_sym_db.RegisterEnumDescriptor(_SCENELAYOUT_LAYOUTTYPE)

_LIGHTINFO_LIGHTTYPE = _descriptor.EnumDescriptor(
    name='LightType',
    full_name='scenenet.LightInfo.LightType',
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name='SPHERE', index=0, number=1,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='PARALLELOGRAM', index=1, number=2,
            options=None,
            type=None),
    ],
    containing_type=None,
    options=None,
    serialized_start=419,
    serialized_end=461,
)
_sym_db.RegisterEnumDescriptor(_LIGHTINFO_LIGHTTYPE)

_INSTANCE_INSTANCETYPE = _descriptor.EnumDescriptor(
    name='InstanceType',
    full_name='scenenet.Instance.InstanceType',
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name='BACKGROUND', index=0, number=1,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='LAYOUT_OBJECT', index=1, number=2,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='LIGHT_OBJECT', index=2, number=3,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='RANDOM_OBJECT', index=3, number=4,
            options=None,
            type=None),
    ],
    containing_type=None,
    options=None,
    serialized_start=1133,
    serialized_end=1219,
)
_sym_db.RegisterEnumDescriptor(_INSTANCE_INSTANCETYPE)

_SCENELAYOUT = _descriptor.Descriptor(
    name='SceneLayout',
    full_name='scenenet.SceneLayout',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='layout_type', full_name='scenenet.SceneLayout.layout_type', index=0,
            number=1, type=14, cpp_type=8, label=1,
            has_default_value=False, default_value=1,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='model', full_name='scenenet.SceneLayout.model', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
        _SCENELAYOUT_LAYOUTTYPE,
    ],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=29,
    serialized_end=195,
)

_LIGHTINFO = _descriptor.Descriptor(
    name='LightInfo',
    full_name='scenenet.LightInfo',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='light_type', full_name='scenenet.LightInfo.light_type', index=0,
            number=1, type=14, cpp_type=8, label=1,
            has_default_value=False, default_value=1,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='light_output', full_name='scenenet.LightInfo.light_output', index=1,
            number=2, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='position', full_name='scenenet.LightInfo.position', index=2,
            number=3, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='radius', full_name='scenenet.LightInfo.radius', index=3,
            number=4, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='v1', full_name='scenenet.LightInfo.v1', index=4,
            number=5, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='v2', full_name='scenenet.LightInfo.v2', index=5,
            number=6, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
        _LIGHTINFO_LIGHTTYPE,
    ],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=198,
    serialized_end=461,
)

_RANDOMOBJECTINFO_TRANSFORMATION = _descriptor.Descriptor(
    name='Transformation',
    full_name='scenenet.RandomObjectInfo.Transformation',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='translation_x', full_name='scenenet.RandomObjectInfo.Transformation.translation_x', index=0,
            number=1, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='translation_y', full_name='scenenet.RandomObjectInfo.Transformation.translation_y', index=1,
            number=2, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='translation_z', full_name='scenenet.RandomObjectInfo.Transformation.translation_z', index=2,
            number=3, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='rotation_mat11', full_name='scenenet.RandomObjectInfo.Transformation.rotation_mat11', index=3,
            number=4, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='rotation_mat12', full_name='scenenet.RandomObjectInfo.Transformation.rotation_mat12', index=4,
            number=5, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='rotation_mat13', full_name='scenenet.RandomObjectInfo.Transformation.rotation_mat13', index=5,
            number=6, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='rotation_mat21', full_name='scenenet.RandomObjectInfo.Transformation.rotation_mat21', index=6,
            number=7, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='rotation_mat22', full_name='scenenet.RandomObjectInfo.Transformation.rotation_mat22', index=7,
            number=8, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='rotation_mat23', full_name='scenenet.RandomObjectInfo.Transformation.rotation_mat23', index=8,
            number=9, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='rotation_mat31', full_name='scenenet.RandomObjectInfo.Transformation.rotation_mat31', index=9,
            number=10, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='rotation_mat32', full_name='scenenet.RandomObjectInfo.Transformation.rotation_mat32', index=10,
            number=11, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='rotation_mat33', full_name='scenenet.RandomObjectInfo.Transformation.rotation_mat33', index=11,
            number=12, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=595,
    serialized_end=896,
)

_RANDOMOBJECTINFO = _descriptor.Descriptor(
    name='RandomObjectInfo',
    full_name='scenenet.RandomObjectInfo',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='shapenet_hash', full_name='scenenet.RandomObjectInfo.shapenet_hash', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='height_meters', full_name='scenenet.RandomObjectInfo.height_meters', index=1,
            number=2, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='object_pose', full_name='scenenet.RandomObjectInfo.object_pose', index=2,
            number=3, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[_RANDOMOBJECTINFO_TRANSFORMATION, ],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=464,
    serialized_end=896,
)

_INSTANCE = _descriptor.Descriptor(
    name='Instance',
    full_name='scenenet.Instance',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='instance_id', full_name='scenenet.Instance.instance_id', index=0,
            number=1, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='semantic_wordnet_id', full_name='scenenet.Instance.semantic_wordnet_id', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='semantic_english', full_name='scenenet.Instance.semantic_english', index=2,
            number=3, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='instance_type', full_name='scenenet.Instance.instance_type', index=3,
            number=4, type=14, cpp_type=8, label=1,
            has_default_value=False, default_value=1,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='light_info', full_name='scenenet.Instance.light_info', index=4,
            number=5, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='object_info', full_name='scenenet.Instance.object_info', index=5,
            number=6, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
        _INSTANCE_INSTANCETYPE,
    ],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=899,
    serialized_end=1219,
)

_POWER = _descriptor.Descriptor(
    name='Power',
    full_name='scenenet.Power',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='r', full_name='scenenet.Power.r', index=0,
            number=1, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='g', full_name='scenenet.Power.g', index=1,
            number=2, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='b', full_name='scenenet.Power.b', index=2,
            number=3, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1221,
    serialized_end=1261,
)

_POSITION = _descriptor.Descriptor(
    name='Position',
    full_name='scenenet.Position',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='x', full_name='scenenet.Position.x', index=0,
            number=1, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='y', full_name='scenenet.Position.y', index=1,
            number=2, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='z', full_name='scenenet.Position.z', index=2,
            number=3, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1263,
    serialized_end=1306,
)

_POSE = _descriptor.Descriptor(
    name='Pose',
    full_name='scenenet.Pose',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='camera', full_name='scenenet.Pose.camera', index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='lookat', full_name='scenenet.Pose.lookat', index=1,
            number=2, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='timestamp', full_name='scenenet.Pose.timestamp', index=2,
            number=3, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1308,
    serialized_end=1405,
)

_VIEW = _descriptor.Descriptor(
    name='View',
    full_name='scenenet.View',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='frame_num', full_name='scenenet.View.frame_num', index=0,
            number=1, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='shutter_open', full_name='scenenet.View.shutter_open', index=1,
            number=2, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='shutter_close', full_name='scenenet.View.shutter_close', index=2,
            number=3, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1407,
    serialized_end=1509,
)

_TRAJECTORY = _descriptor.Descriptor(
    name='Trajectory',
    full_name='scenenet.Trajectory',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='layout', full_name='scenenet.Trajectory.layout', index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='instances', full_name='scenenet.Trajectory.instances', index=1,
            number=2, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='views', full_name='scenenet.Trajectory.views', index=2,
            number=3, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='render_path', full_name='scenenet.Trajectory.render_path', index=3,
            number=4, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1512,
    serialized_end=1654,
)

_TRAJECTORIES = _descriptor.Descriptor(
    name='Trajectories',
    full_name='scenenet.Trajectories',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='trajectories', full_name='scenenet.Trajectories.trajectories', index=0,
            number=1, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1656,
    serialized_end=1714,
)

_SCENELAYOUT.fields_by_name['layout_type'].enum_type = _SCENELAYOUT_LAYOUTTYPE
_SCENELAYOUT_LAYOUTTYPE.containing_type = _SCENELAYOUT
_LIGHTINFO.fields_by_name['light_type'].enum_type = _LIGHTINFO_LIGHTTYPE
_LIGHTINFO.fields_by_name['light_output'].message_type = _POWER
_LIGHTINFO.fields_by_name['position'].message_type = _POSITION
_LIGHTINFO.fields_by_name['v1'].message_type = _POSITION
_LIGHTINFO.fields_by_name['v2'].message_type = _POSITION
_LIGHTINFO_LIGHTTYPE.containing_type = _LIGHTINFO
_RANDOMOBJECTINFO_TRANSFORMATION.containing_type = _RANDOMOBJECTINFO
_RANDOMOBJECTINFO.fields_by_name['object_pose'].message_type = _RANDOMOBJECTINFO_TRANSFORMATION
_INSTANCE.fields_by_name['instance_type'].enum_type = _INSTANCE_INSTANCETYPE
_INSTANCE.fields_by_name['light_info'].message_type = _LIGHTINFO
_INSTANCE.fields_by_name['object_info'].message_type = _RANDOMOBJECTINFO
_INSTANCE_INSTANCETYPE.containing_type = _INSTANCE
_POSE.fields_by_name['camera'].message_type = _POSITION
_POSE.fields_by_name['lookat'].message_type = _POSITION
_VIEW.fields_by_name['shutter_open'].message_type = _POSE
_VIEW.fields_by_name['shutter_close'].message_type = _POSE
_TRAJECTORY.fields_by_name['layout'].message_type = _SCENELAYOUT
_TRAJECTORY.fields_by_name['instances'].message_type = _INSTANCE
_TRAJECTORY.fields_by_name['views'].message_type = _VIEW
_TRAJECTORIES.fields_by_name['trajectories'].message_type = _TRAJECTORY
DESCRIPTOR.message_types_by_name['SceneLayout'] = _SCENELAYOUT
DESCRIPTOR.message_types_by_name['LightInfo'] = _LIGHTINFO
DESCRIPTOR.message_types_by_name['RandomObjectInfo'] = _RANDOMOBJECTINFO
DESCRIPTOR.message_types_by_name['Instance'] = _INSTANCE
DESCRIPTOR.message_types_by_name['Power'] = _POWER
DESCRIPTOR.message_types_by_name['Position'] = _POSITION
DESCRIPTOR.message_types_by_name['Pose'] = _POSE
DESCRIPTOR.message_types_by_name['View'] = _VIEW
DESCRIPTOR.message_types_by_name['Trajectory'] = _TRAJECTORY
DESCRIPTOR.message_types_by_name['Trajectories'] = _TRAJECTORIES

SceneLayout = _reflection.GeneratedProtocolMessageType('SceneLayout', (_message.Message,), dict(
    DESCRIPTOR=_SCENELAYOUT,
    __module__='scenenet_pb2'
    # @@protoc_insertion_point(class_scope:scenenet.SceneLayout)
))
_sym_db.RegisterMessage(SceneLayout)

LightInfo = _reflection.GeneratedProtocolMessageType('LightInfo', (_message.Message,), dict(
    DESCRIPTOR=_LIGHTINFO,
    __module__='scenenet_pb2'
    # @@protoc_insertion_point(class_scope:scenenet.LightInfo)
))
_sym_db.RegisterMessage(LightInfo)

RandomObjectInfo = _reflection.GeneratedProtocolMessageType('RandomObjectInfo', (_message.Message,), dict(
    
    Transformation=_reflection.GeneratedProtocolMessageType('Transformation', (_message.Message,), dict(
        DESCRIPTOR=_RANDOMOBJECTINFO_TRANSFORMATION,
        __module__='scenenet_pb2'
        # @@protoc_insertion_point(class_scope:scenenet.RandomObjectInfo.Transformation)
    )),
    DESCRIPTOR=_RANDOMOBJECTINFO,
    __module__='scenenet_pb2'
    # @@protoc_insertion_point(class_scope:scenenet.RandomObjectInfo)
))
_sym_db.RegisterMessage(RandomObjectInfo)
_sym_db.RegisterMessage(RandomObjectInfo.Transformation)

Instance = _reflection.GeneratedProtocolMessageType('Instance', (_message.Message,), dict(
    DESCRIPTOR=_INSTANCE,
    __module__='scenenet_pb2'
    # @@protoc_insertion_point(class_scope:scenenet.Instance)
))
_sym_db.RegisterMessage(Instance)

Power = _reflection.GeneratedProtocolMessageType('Power', (_message.Message,), dict(
    DESCRIPTOR=_POWER,
    __module__='scenenet_pb2'
    # @@protoc_insertion_point(class_scope:scenenet.Power)
))
_sym_db.RegisterMessage(Power)

Position = _reflection.GeneratedProtocolMessageType('Position', (_message.Message,), dict(
    DESCRIPTOR=_POSITION,
    __module__='scenenet_pb2'
    # @@protoc_insertion_point(class_scope:scenenet.Position)
))
_sym_db.RegisterMessage(Position)

Pose = _reflection.GeneratedProtocolMessageType('Pose', (_message.Message,), dict(
    DESCRIPTOR=_POSE,
    __module__='scenenet_pb2'
    # @@protoc_insertion_point(class_scope:scenenet.Pose)
))
_sym_db.RegisterMessage(Pose)

View = _reflection.GeneratedProtocolMessageType('View', (_message.Message,), dict(
    DESCRIPTOR=_VIEW,
    __module__='scenenet_pb2'
    # @@protoc_insertion_point(class_scope:scenenet.View)
))
_sym_db.RegisterMessage(View)

Trajectory = _reflection.GeneratedProtocolMessageType('Trajectory', (_message.Message,), dict(
    DESCRIPTOR=_TRAJECTORY,
    __module__='scenenet_pb2'
    # @@protoc_insertion_point(class_scope:scenenet.Trajectory)
))
_sym_db.RegisterMessage(Trajectory)

Trajectories = _reflection.GeneratedProtocolMessageType('Trajectories', (_message.Message,), dict(
    DESCRIPTOR=_TRAJECTORIES,
    __module__='scenenet_pb2'
    # @@protoc_insertion_point(class_scope:scenenet.Trajectories)
))
_sym_db.RegisterMessage(Trajectories)

# @@protoc_insertion_point(module_scope)
