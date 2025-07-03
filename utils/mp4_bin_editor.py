import struct
from typing import Dict, Tuple, Optional, Union

class MP4BinEditor:
    def __init__(self, mp4_data: bytes):
        self.mp4_data = bytearray(mp4_data)
        self.box_positions: Dict[str, Tuple[int, int]] = {}  # {box_type: (start_offset, end_offset)}
        self._parse_mp4(0, len(mp4_data))

    def _parse_mp4(self, offset: int, max_offset: int) -> None:
        """Recursively parse MP4 boxes and record their positions."""
        recur_types = {'moov', 'trak', 'mdia', 'minf', 'stbl', 'moof', 'traf'}

        while offset < len(self.mp4_data):
            if max_offset is not None and offset + 8 >= max_offset:
                return
            box_size, box_type = self._read_box_header(offset)

            # Record the first occurrence of each box type
            if box_type not in self.box_positions:
                self.box_positions[box_type] = (offset, offset + box_size)

            # Recursively parse container boxes
            if box_type in recur_types:
                self._parse_mp4(offset + 8, offset + box_size)
            offset += box_size

    def _read_box_header(self, offset: int) -> Tuple[int, str]:
        """Read box header at given offset."""
        box_header = self.mp4_data[offset:offset + 8]
        box_size, box_type = struct.unpack('>I4s', box_header)
        return box_size, box_type.decode('utf-8')

    def _get_box_data(self, box_type: str) -> Tuple[int, int]:
        """Get the full data of a box (including header)."""
        if box_type not in self.box_positions:
            raise ValueError(f"Box {box_type} not found")
        return self.box_positions[box_type]

    def _modify_box_value(self, box_type: str, value_offset: int, value: int, value_size: int) -> None:
        """Internal method to modify a value within a box."""
        if box_type not in self.box_positions:
            raise ValueError(f"Box {box_type} not found")

        box_start, _ = self.box_positions[box_type]
        actual_offset = box_start + value_offset

        if actual_offset + value_size > len(self.mp4_data):
            raise ValueError("Invalid offset or value size")

        if value_size == 4:
            packed = struct.pack('>I', value)
        elif value_size == 8:
            packed = struct.pack('>Q', value)
        else:
            raise ValueError("Unsupported value size")

        self.mp4_data[actual_offset:actual_offset + value_size] = packed

    # Specific box value accessors
    def sequence_number(self, value: Optional[int] = None) -> Optional[int]:
        """Get or set the sequence number in mfhd box."""
        if value is None:
            # Getter
            start, _ = self._get_box_data('mfhd')
            return struct.unpack_from('>I', self.mp4_data, start + 20)[0]
        else:
            # Setter
            self._modify_box_value('mfhd', 20, value, 4)

    def base_media_decode_time(self, value: Optional[int] = None) -> Optional[int]:
        """Get or set the base media decode time in tfdt box."""
        start, _ = self._get_box_data('tfdt')

        version = self.mp4_data[start + 8]
        value_size = 8 if version == 1 else 4

        if value is None:
            # Getter
            return struct.unpack_from('>Q' if version == 1 else '>I', self.mp4_data, start + 12)[0]
        else:
            # Setter
            self._modify_box_value('tfdt', 12, value, value_size)

    def mdhd_timescale(self, value: Optional[int] = None) -> Optional[int]:
        """Get or set the timescale in mdhd box."""
        start, _ = self._get_box_data('mdhd')

        version = self.mp4_data[start + 8]
        value_offset = 20 + version * 8

        if value is None:
            # Getter
            return struct.unpack_from('>I', self.mp4_data, start + value_offset)[0]
        else:
            # Setter
            self._modify_box_value('mdhd', value_offset, value, 4)

    def sidx_earliest_presentation_time(self, value: Optional[int] = None) -> Optional[int]:
        """Get or set the earliest presentation time in sidx box."""
        start, _ = self._get_box_data('sidx')

        version = self.mp4_data[start + 8]
        value_size = 8 if version != 0 else 4

        if value is None:
            # Getter
            return struct.unpack_from('>Q' if version != 0 else '>I', self.mp4_data, start + 20)[0]
        else:
            # Setter
            self._modify_box_value('sidx', 20, value, value_size)

    def export(self) -> bytes:
        """Export the modified MP4 data."""
        return bytes(self.mp4_data)

def update_reencode_metadata(old_bytes: bytes, new_bytes: bytes) -> bytes:
    old_editor = MP4BinEditor(old_bytes)
    new_editor = MP4BinEditor(new_bytes)

    new_timescale = new_editor.mdhd_timescale()
    begin_time = old_editor.base_media_decode_time() // old_editor.mdhd_timescale()
    new_editor.sequence_number(old_editor.sequence_number())
    new_editor.base_media_decode_time(begin_time * new_timescale)
    new_editor.sidx_earliest_presentation_time(begin_time * new_timescale)
    return new_editor.export()
