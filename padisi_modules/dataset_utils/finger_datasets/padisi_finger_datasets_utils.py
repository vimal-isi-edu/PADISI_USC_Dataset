from collections import namedtuple
from typing import Callable

import cv2
import numpy as np

from padisi_modules.dataIO.padisi_dataset import PadisiH5Reader, PadisiH5ReaderDataExtractor
from padisi_modules.dataset_utils.collection_attributes import CollectionAttributes, guess_collection_attribute_set
from padisi_modules.utils.attribute_validation import validate_data_attributes
from padisi_modules.utils.unique_identifier_utils import get_object_str_identifier

SpecRangeDataSetAttrs = namedtuple("SpecRangeDataSetAttrs", ['light_ds', 'dark_ds', 'lambdas', 'trans_to_canon',
                                                             'bit_depth'])
SpecRangeExtractAttrs = namedtuple("SpecRangeExtractAttrs",
                                   ['name', 'lambdas', 'roi', 'lit_frames', 'dark_frames', 'roi_relative_to'])
SpecRangeExtractAttrs.__new__.__defaults__ = (None, None, None, None, None, 'canon')


# Use wrapper function with default arguments to avoid warning of namedtuple fields not being initialized.
def spec_range_extract_attrs(name: str, lambdas: (None, tuple, list, np.ndarray) = None,
                             roi: (None, dict) = None, lit_frames: (None, tuple, list, np.ndarray) = None,
                             dark_frames: (None, tuple, list, np.ndarray) = None,
                             roi_relative_to: str = 'canon'):

    return SpecRangeExtractAttrs(name=name, lambdas=lambdas, roi=roi, lit_frames=lit_frames, dark_frames=dark_frames,
                                 roi_relative_to=roi_relative_to)


def get_finger_spec_range_attrs(attr, spec_range: str):
    if spec_range.lower() == 'swir':
        return SpecRangeDataSetAttrs(attr.swir_light_datasets, attr.swir_dark_datasets, attr.SWIR_LED_LAMBDAS,
                                     attr.SWIR_FINGER_SLIT_TO_CANONICAL,
                                     attr.get_data_bit_depth('FINGER_READER_LS'))
    elif spec_range.lower() == 'nir':
        return SpecRangeDataSetAttrs(attr.nir_light_finger_datasets, attr.nir_dark_finger_datasets,
                                     attr.NIR_LED_LAMBDAS_FINGER, attr.NIR_FINGER_SLIT_TO_CANONICAL,
                                     attr.get_data_bit_depth('FINGER_READER_VNB'))
    elif spec_range.lower() == 'vis':
        return SpecRangeDataSetAttrs(attr.vis_light_finger_datasets, attr.vis_dark_finger_datasets,
                                     attr.VIS_LED_LAMBDAS_FINGER, attr.VIS_FINGER_SLIT_TO_CANONICAL,
                                     attr.get_data_bit_depth('FINGER_READER_VNB'))
    elif spec_range.lower() == 'lsci':
        return SpecRangeDataSetAttrs(attr.lsci_light_datasets, attr.lsci_dark_datasets,
                                     attr.LSCI_LED_LAMBDAS, attr.LSCI_FINGER_SLIT_TO_CANONICAL,
                                     attr.get_data_bit_depth('FINGER_READER_LS'))
    elif spec_range.lower() == 'bi':
        return SpecRangeDataSetAttrs(attr.bi_light_finger_datasets, attr.bi_dark_finger_datasets,
                                     attr.BI_LED_LAMBDAS, attr.VIS_FINGER_SLIT_TO_CANONICAL,
                                     attr.get_data_bit_depth('FINGER_READER_VNB'))
    else:
        raise RuntimeError("{} Error: Unidentified spectral range '{}' for finger data".
                           format("get_finger_spec_range_attrs()", spec_range))


FINGER_DATA_EXTRACTION_MODES = {"light_only", "dark_only", "diff", "concat"}
FINGER_SPEC_RANGES = ('lsci', 'swir', 'vis', 'nir', 'bi')


class FingerSlitTransformerAndROICropper(object):
    def __init__(self, attr: CollectionAttributes, source_to_can_map: dict,
                 auto_canon_target: bool = True, target_finger_slit_size: (None, dict) = None,
                 crop_roi: bool = True, roi: (None, dict) = None, roi_relative_to: str ='canon'):
        """
        :param attr                   : CollectionAttributes, collection event attributes
        :param source_to_can_map      : dict, mapping parameters from source data to canonical size/view.
        :param auto_canon_target      : bool, automatically set target to canonical if the target size is not provided.
        :param target_finger_slit_size: dict, target finger slit size before cropping roi.
        :param crop_roi               : bool, whether to crop or ROI or not.
        :param roi                    : dict, roi area coordinates of the sample image.
        :param roi_relative_to        : str, the size to which the ROI is relative, must be
                                          * 'source', which is the finger slit image after being cropped and rotated
                                            to canonical pose
                                          * 'target', which the target size of the finger slit after being resize from
                                            source size to target size
                                          * 'canon', which is the canonical finger slit size
                                        In all cases, the roi is assumed to be in relation to the finger slit image
                                        after being transformed to be in the same pose as the canonical finger
                                        slit pose.
        """
        assert not crop_roi or (roi is not None and
                                isinstance(roi, dict) and
                                roi.keys() >= {'x1', 'x2', 'y1', 'y2'} and
                                roi_relative_to in {'source', 'target', 'canon'})
        if target_finger_slit_size is None and auto_canon_target:
            target_finger_slit_size = attr.CANONICAL_FINGER_SIZE
        self.source_to_can_map = source_to_can_map
        self.target_finger_slit_size = target_finger_slit_size
        self.attr = attr
        self.crop_roi = crop_roi
        self.roi = roi
        self.roi_relative_to = roi_relative_to

    def __call__(self, reader: PadisiH5Reader, datasets: (tuple, list),
                 frames: (None, tuple, list) = None, avg_frames: bool = True) -> np.ndarray:
        """
        Generate crops for finger images.

        :param reader    : PadisiH5Reader containing numeric arrays of all biometric modalities images.
        :param datasets  : list(str), list of basler dataset attributes/filters, e.g., attr.basler_light_datasets.
        :param frames    : list, indices of frames to collect from a video sequence.
        :param avg_frames: bool, whether to average frames or not.
        :return          : numpy ndarray, 4D array of frames x channels x height x width of cropped images.
        """

        # get source images from reader
        images = [reader.get_dataset_frames(dataset_name=ds, frames=frames) for ds in datasets]
        # average frames
        if avg_frames:
            images = [np.mean(img, axis=0, keepdims=True) for img in images]
        # crop finger slit first
        fin = self.source_to_can_map['crop']
        images = [img[:, :, fin['y1']:fin['y2'], fin['x1']:fin['x2']] for img in images]
        # rotate to canonical view
        if self.source_to_can_map['rot90'] != 0:
            images = [np.rot90(m=img, k=self.source_to_can_map['rot90'], axes=(2, 3)) for img in images]
        # flip left to right
        if self.source_to_can_map['fliplr']:
            images = [img[:, :, :, ::-1] for img in images]
        # resize to target size
        source_w, source_h = images[0].shape[3], images[0].shape[2]
        source_sx = 1.0 if self.target_finger_slit_size is None else self.target_finger_slit_size['width'] / source_w
        source_sy = 1.0 if self.target_finger_slit_size is None else self.target_finger_slit_size['height'] / source_h
        if source_sx != 1.0 or source_sy != 1.0:
            images_copy = images
            images = []
            for i, img in enumerate(images_copy):
                n_frames = img.shape[0]
                n_channels = img.shape[1]
                res_image = np.empty((n_frames, n_channels,
                                      self.target_finger_slit_size['height'],
                                      self.target_finger_slit_size['width']))
                for f in range(n_frames):
                    for c in range(n_channels):
                        res_image[f, c, :, :] = cv2.resize(img[f, c, :, :], None,
                                                           fx=source_sx, fy=source_sy,
                                                           interpolation=cv2.INTER_CUBIC)
                images.append(res_image)

        # stack images
        images = np.concatenate(images, axis=1)

        if self.crop_roi:
            roi = self.roi.copy()
            if self.roi_relative_to == 'target':
                roi_sx = roi_sy = 1.0
            elif self.roi_relative_to in {'canon', 'source'}:
                # scale roi to target size
                base_w = self.attr.CANONICAL_FINGER_SIZE['width'] if self.roi_relative_to == 'canon' else source_w
                base_h = self.attr.CANONICAL_FINGER_SIZE['height'] if self.roi_relative_to == 'canon' else source_h
                roi_sx = images.shape[3] / base_w
                roi_sy = images.shape[2] / base_h
            else:
                assert False, "Possible Bug: Unsupported roi_relative_to value" % self.roi_relative_to

            scaled_roi = {'x1': int(roi['x1'] * roi_sx), 'x2': int(roi['x2'] * roi_sx),
                          'y1': int(roi['y1'] * roi_sy), 'y2': int(roi['y2'] * roi_sy)}

            images = images[:, :, scaled_roi['y1']:scaled_roi['y2'], scaled_roi['x1']:scaled_roi['x2']]
        images = images.astype('float32')
        return images


class FingerSingleSpecRangePadisiH5ReaderDataExtractor(PadisiH5ReaderDataExtractor):
    VALID_TARGET_SLIT_KEYS = ('width', 'height')
    VALID_ROI_KEYS = ('x1', 'x2', 'y1', 'y2')
    VALID_ROI_RELATIVE_TO = ('source', 'target', 'canon')

    def get_property_extractors_validation_dictionary(self) -> (None, Callable, dict):
        return None

    def __init__(self, spec_range: str, lambdas: (None, tuple, list, np.ndarray) = None, crop_roi: bool = True,
                 roi: (None, dict) = None, roi_relative_to: str = 'canon',
                 auto_canon_target: bool = True, target_slit_size: (None, dict) = None, mode: str = 'diff',
                 lit_frames: (None, tuple, list, np.ndarray) = None,
                 dark_frames: (None, tuple, list, np.ndarray) = None,
                 bit_depth_norm: bool = False):
        """
        Initializes finger data extractor.

        :param spec_range       : str, one of the allowed spectral ranges for the finger as described in
                                  FINGER_SPEC_RANGES.
        :param lambdas          : list, list of integer-valued wavelengths to be collected for the specified spectral
                                  range. If None, all available wavelengths are used. Note that wavelengths are
                                  automatically sorted by wavelength number - with the exception of white (number 0)
                                  which always comes last.
                      (Optional)  Default is None.
        :param crop_roi         : bool, whether to crop ROI or not.
                      (Optional)  Default is True.
        :param roi              : dict, the region of interest to be cropped.
                      (Optional)  Default is None.
        :param roi_relative_to  : str, to what is the provided ROI is relative, must be:
                                     * 'source', which is the finger slit image after being cropped and rotated to
                                        canonical pose.
                                     * 'target', which the target size of the finger slit after being resized from
                                        source size to target size.
                                     * 'canon', which is the canonical finger slit size.

                                     In all cases, the roi is assumed to be in relation to the finger slit image after
                                     being transformed to be in the same pose as the canonical finger slit pose. If roi
                                     is not provided, this field is ignored and 'canon' is assumed.
                      (Optional)  Default is 'canon'.
        :param auto_canon_target: bool, automatically set target to canonical if the target size is not provided.
                      (Optional)  Default is True.
        :param target_slit_size : dict, with two keys 'width' and 'height' for the size of the target finger slit for
                                  this extractor.
                      (Optional)  Default is None.
        :param mode             : str, one of the modes in FINGER_DATA_EXTRACTION_MODES, which determine the way to
                                  handle light and dark frames.
                      (Optional)  Default is 'diff'.
        :param lit_frames       : tuple/list/np.ndarray, sequence of integer indices for the frames to use from a light
                                  dataset. If None, all frames are used.
                      (Optional)  Default is None.
        :param dark_frames      : tuple/list/np.ndarray, sequence of integer indices for the frames to use from a
                                  dark dataset. If None, all frames are used.
                      (Optional)  Default is None.
        :param bit_depth_norm   : bool, apply bit-depth normalization to the range [0, 1].
                      (Optional)  Default is False.

        Note that the order of the 'lambdas' is irrelevant and the wavelengths are always sorted based on their actual
        values.
        """
        super(FingerSingleSpecRangePadisiH5ReaderDataExtractor, self).__init__()
        self.spec_range = spec_range
        self.lambdas = lambdas
        self.crop_roi = crop_roi
        self.roi = roi
        self.roi_relative_to = roi_relative_to
        self.auto_canon_target = auto_canon_target
        self.target_slit_size = target_slit_size
        self.mode = mode
        self.lit_frames = lit_frames
        self.dark_frames = dark_frames
        self.bit_depth_norm = bit_depth_norm

    def __str__(self):
        return get_object_str_identifier(self)

    def _extract_padisi_h5_data(self, padisi_h5_reader: PadisiH5Reader) -> np.ndarray:
        attr = guess_collection_attribute_set(padisi_h5_reader)
        spec_range_attrs = get_finger_spec_range_attrs(attr, self.spec_range)

        # copy the lambdas locally because they may be updated here, and this will change the
        # extractor object's status, which will affect its identifier string
        lambdas = self.lambdas
        if lambdas is None:
            lambdas = spec_range_attrs.lambdas
            # Sort wavelengths with special handling of 0 valued wavelength (white) - always put last.
            lambdas = [lambdas[i] for i in np.argsort([np.inf if item == 0 else item for item in lambdas])]
        try:
            lambdas_idx = [spec_range_attrs.lambdas.index(lm) for lm in lambdas]
        except ValueError as e:
            raise ValueError("{} Error: At least one of the requested wavelengths ({}) is not in the available "
                             "wavelengths in this dataset ({}): {}".
                             format(self.__class__.__name__ + "._extract_padisi_h5_data()",
                                    ', '.join([str(item) for item in lambdas]),
                                    ', '.join([str(item) for item in spec_range_attrs.lambdas]), str(e)))

        if self.bit_depth_norm:
            assert spec_range_attrs.bit_depth is not None, \
                "{} Error: When 'bit_depth_norm' = True, bit_depth value from test attributes cannot be " \
                "None.".format(self.__class__.__name__ + "._extract_padisi_h5_data()")
            validate_data_attributes([spec_range_attrs.bit_depth], (np.integer,), ('scalar', '>', 0),
                                     var_name='spec_range_attrs.bit_depth',
                                     func_name=self.__class__.__name__ + "._extract_padisi_h5_data()")

        light_ds = spec_range_attrs.light_ds()
        light_ds = [light_ds[i][1:] for i in lambdas_idx]
        dark_ds = spec_range_attrs.dark_ds()
        if len(dark_ds) > 0:
            dark_ds = [dark_ds[i][1:] for i in lambdas_idx]
        roi = self.roi
        roi_relative_to = self.roi_relative_to
        if self.crop_roi and self.roi is None:
            roi = attr.CANONICAL_FINGER_ROI

            roi_relative_to = 'canon'
        tran_crop_func = FingerSlitTransformerAndROICropper(attr, spec_range_attrs.trans_to_canon,
                                                            auto_canon_target=self.auto_canon_target,
                                                            target_finger_slit_size=self.target_slit_size,
                                                            crop_roi=self.crop_roi, roi=roi,
                                                            roi_relative_to=roi_relative_to,
                                                            )

        # Create light and dark crops and combine them based on the chosen mode
        lit_crops = dark_crops = None
        if self.mode != 'dark_only':
            lit_crops = tran_crop_func(padisi_h5_reader, light_ds, frames=self.lit_frames, avg_frames=False)
            if self.bit_depth_norm:
                lit_crops = lit_crops / (2**spec_range_attrs.bit_depth - 1)
                lit_crops[lit_crops > 1.0] = 1.0
        if self.mode != 'light_only' and len(dark_ds) > 0:
            dark_crops = tran_crop_func(padisi_h5_reader, dark_ds, frames=self.dark_frames, avg_frames=True)
            if self.bit_depth_norm:
                dark_crops = dark_crops / (2**spec_range_attrs.bit_depth - 1)
                dark_crops[dark_crops > 1.0] = 1.0
        if self.mode == 'light_only' or (len(dark_ds) == 0 and (self.mode == 'diff' or self.mode == 'concat')):
            return lit_crops
        if self.mode == 'dark_only':
            return dark_crops
        # At this point, we are sure that there are dark datasets for this required spectral range
        assert dark_crops is not None, \
            "{} Error: Possible bug, dark_crops must be available at this point.".format(self.__class__.__name__)
        assert dark_crops.shape[1] == lit_crops.shape[1], \
            "{} Error: Dark and light data of the same spectral range must have the same number of " \
            "channels.".format(self.__class__.__name__ + "._extract_padisi_h5_data()")

        if self.mode == 'diff':
            diff_crops = lit_crops - dark_crops
            diff_crops[np.where(diff_crops < 0)] = 0
            return diff_crops
        if self.mode == 'concat':
            if lit_crops.shape[0] > 1:
                # If light dataset has multiple frames, then, concatenate over frames.
                # We are sure that dark always has one frame because of frame averaging above
                concat_crops = np.concatenate((lit_crops, dark_crops), axis=0)
            else:
                # Else, concatenate over channels
                concat_crops = np.concatenate((lit_crops, dark_crops), axis=1)
            return concat_crops

        raise RuntimeError("{} Error: Potential Bug: Mode '{}' is not recognized among allowed options: {}".format(
            self.__class__.__name__ + "._extract_padisi_h5_data()", self.mode, FINGER_DATA_EXTRACTION_MODES))

    def _validate_frames(self, frames: (tuple, list), var_name: str) -> list:
        assert isinstance(var_name, str) and len(var_name) > 0, \
            "{} Error: Variable 'var_name' must be a non-empty string.".format(self.__class__.__name__)

        frames = validate_data_attributes(frames, (np.integer,), ('nonempty', 'vector', '>=', 0),
                                          var_name=var_name, func_name=self.__class__.__name__ + "._validate_frames()")

        return sorted([int(item) for item in frames])

    @property
    def spec_range(self) -> str:
        return self._spec_range

    @spec_range.setter
    def spec_range(self, spec_range: str) -> None:
        assert isinstance(spec_range, str) and len(spec_range) > 0, \
            "{} Error: Property 'spec_range' must be a non-empty string.".format(self.__class__.__name__)
        assert spec_range.lower() in FINGER_SPEC_RANGES, \
            "{} Error: Property 'spec_range' = '{}' is invalid. Valid spectral ranges are: " \
            "[{}].".format(self.__class__.__name__, spec_range,
                           ', '.join(["'" + item + "'" for item in FINGER_SPEC_RANGES]))
        self._spec_range = spec_range.lower()

    @property
    def lambdas(self) -> (None, tuple):
        return self._lambdas

    @lambdas.setter
    def lambdas(self, lambdas: (None, tuple, list, np.ndarray)) -> None:
        if lambdas is not None:
            lambdas = validate_data_attributes(lambdas, (np.integer,), ('vector', 'nonempty', '>=', 0),
                                               var_name='lambdas', func_name=self.__class__.__name__)
            lambdas = [int(item) for item in lambdas]
            # Sort wavelengths with special handling of 0 valued wavelength (white) - always put last.
            lambdas = [lambdas[i] for i in np.argsort([np.inf if item == 0 else item for item in lambdas])]
        self._lambdas = lambdas

    @property
    def crop_roi(self) -> bool:
        return self._crop_roi

    @crop_roi.setter
    def crop_roi(self, crop_roi: bool) -> None:
        assert isinstance(crop_roi, bool), \
            "{} Error: Property 'crop_roi' must be boolean.".format(self.__class__.__name__)
        self._crop_roi = crop_roi

    @property
    def roi(self) -> (None, dict):
        return self._roi

    @roi.setter
    def roi(self, roi: (None, dict)) -> None:
        if roi is not None:
            assert isinstance(roi, dict) and len(roi) == 4, \
                "{} Error: Property 'roi' must be a dictionary with 4 elements.".format(self.__class__.__name__)

            missing_keys = [item for item in FingerSingleSpecRangePadisiH5ReaderDataExtractor.VALID_ROI_KEYS if item not
                            in roi.keys()]
            assert len(missing_keys) == 0, \
                "{} Error: Property 'roi' is missing keys [{}]. Valid keys are " \
                "[{}].".format(self.__class__.__name__, ', '.join(["'" + item + "'" for item in missing_keys]),
                               ', '.join(["'" + item + "'" for item in
                                          FingerSingleSpecRangePadisiH5ReaderDataExtractor.VALID_ROI_KEYS]))
            try:
                validate_data_attributes(list(roi.values()), (np.integer,), ('vector', '>=', 0))
            except (ValueError, AssertionError):
                raise RuntimeError("{} Error: Values of dictionary 'roi' must be non-negative integers.".
                                   format(self.__class__.__name__))
        self._roi = roi

    @property
    def roi_relative_to(self) -> str:
        return self._roi_relative_to

    @roi_relative_to.setter
    def roi_relative_to(self, roi_relative_to: str) -> None:
        assert isinstance(roi_relative_to, str) and len(roi_relative_to) > 0, \
            "{} Error: Property 'roi_relative_to' must be a non-empty string.".format(self.__class__.__name__)

        assert roi_relative_to.lower() in FingerSingleSpecRangePadisiH5ReaderDataExtractor.VALID_ROI_RELATIVE_TO, \
            "{} Error: Property 'roi_relative_to' = '{}' is invalid. Valid values are " \
            "[{}].".format(self.__class__.__name__, roi_relative_to,
                           ', '.join(["'" + item + "'" for item in
                                      FingerSingleSpecRangePadisiH5ReaderDataExtractor.VALID_ROI_RELATIVE_TO]))
        self._roi_relative_to = roi_relative_to.lower()

    @property
    def auto_canon_target(self) -> bool:
        return self._auto_canon_target

    @auto_canon_target.setter
    def auto_canon_target(self, auto_canon_target: bool) -> None:
        assert isinstance(auto_canon_target, bool), \
            "{} Error: Property 'crop_roi' must be boolean.".format(self.__class__.__name__)
        self._auto_canon_target = auto_canon_target

    @property
    def target_slit_size(self) -> (None, dict):
        return self._target_slit_size

    @target_slit_size.setter
    def target_slit_size(self, target_slit_size: (None, dict)) -> None:
        if target_slit_size is not None:
            assert isinstance(target_slit_size, dict) and len(target_slit_size) == 2, \
                "{} Error: Property 'target_slit_size' must be a dictionary with 2 " \
                "elements.".format(self.__class__.__name__)

            missing_keys = [item for item in FingerSingleSpecRangePadisiH5ReaderDataExtractor.VALID_TARGET_SLIT_KEYS if
                            item not in target_slit_size.keys()]
            assert len(missing_keys) == 0, \
                "{} Error: Property 'target_slit_size' is missing keys [{}]. Valid keys are " \
                "[{}]".format(self.__class__.__name__, ', '.join(["'" + item + "'" for item in missing_keys]),
                              ', '.join(["'" + item + "'" for item in
                                         FingerSingleSpecRangePadisiH5ReaderDataExtractor.VALID_TARGET_SLIT_KEYS]))

            try:
                validate_data_attributes(list(target_slit_size.values()), (np.integer,), ('vector', '>', 0))
            except (ValueError, AssertionError):
                raise RuntimeError("{} Error: Values of dictionary 'target_slit_size' must be positive integers.".
                                   format(self.__class__.__name__))
        self._target_slit_size = target_slit_size

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        assert isinstance(mode, str) and len(mode) > 0, \
            "{} Error: Property 'mode' must be a non-empty string.".format(self.__class__.__name__)
        assert mode.lower() in FINGER_DATA_EXTRACTION_MODES, \
            "{} Error: Property 'mode' = '{}' is invalid. Valid modes are: " \
            "[{}].". format(self.__class__.__name__, mode,
                            ', '.join(["'" + item + "'" for item in FINGER_DATA_EXTRACTION_MODES]))
        self._mode = mode.lower()

    @property
    def lit_frames(self) -> (None, list):
        return self._lit_frames

    @lit_frames.setter
    def lit_frames(self, lit_frames: (None, tuple, list)) -> None:
        if lit_frames is not None:
            # Make sure frames are returned sorted.
            lit_frames = self._validate_frames(lit_frames, 'lit_frames')
        self._lit_frames = lit_frames

    @property
    def dark_frames(self) -> (None, list):
        return self._dark_frames

    @dark_frames.setter
    def dark_frames(self, dark_frames: (None, tuple, list)) -> None:
        if dark_frames is not None:
            # Make sure frames are returned sorted.
            dark_frames = self._validate_frames(dark_frames, 'dark_frames')
        self._dark_frames = dark_frames

    @property
    def bit_depth_norm(self) -> bool:
        return self._bit_depth_norm

    @bit_depth_norm.setter
    def bit_depth_norm(self, bit_depth_norm: bool) -> None:
        assert isinstance(bit_depth_norm, bool), \
            "{} Error: Property 'bit_depth_norm' must be boolean.".format(self.__class__.__name__)
        self._bit_depth_norm = bit_depth_norm


class FingerMultiSpecRangePadisiH5ReaderDataExtractor(PadisiH5ReaderDataExtractor):
    def __init__(self, spec_ranges_extract_attrs: (tuple, list), crop_roi: bool = True, auto_canon_target: bool = True,
                 target_slit_size: dict = None, mode: str = 'diff', concat_axis: (int, np.ndarray) = 1,
                 bit_depth_norm: bool = False):
        """
        Combines multiple single spectral range data and concatenate them to create a single data cube.

        :param spec_ranges_extract_attrs: list, of attributes for single spectral ranges as type SpecRangeExtractAttrs
        :param crop_roi                 : bool, whether to crop an ROI or not.
                              (Optional)  Default is True.
        :param auto_canon_target        : bool, automatically set target to canonical if the target size is not provided
                              (Optional)  Default is True.
        :param target_slit_size         : dict, with two keys 'width' and 'height' for the size of the target finger
                                          slit for all extractors.
                              (Optional)  Default is None.
        :param mode                     : str, the same as the mode argument for the
                                          FingerSingleSpecRangePadisiH5ReaderDataExtractor class.
                              (Optional)  Default is 'diff'.
        :param concat_axis              : int (0 or 1), the axis on which all data cubes are concatenated, all data
                                          cubes must be 4D, and the axis of concatenation is either the first or
                                          second, in both cases, each cube is reshaped so that the other axis (the one
                                          of the first two that is not used for concatenation) is 1, which either
                                          converts channels into frames or vice versa, so that the concatenation
                                          operation is always valid.
                              (Optional)  Default is 1.

        Note that the order of the 'spec_ranges_extract_attrs' is irrelevant and the spectral channels as well as their
        wavelengths are always sorted based on the sequence ('lsci', 'swir', 'vis', 'nir', 'bi') and the corresponding
        value of the wavelengths.
        """
        super(FingerMultiSpecRangePadisiH5ReaderDataExtractor, self).__init__()

        assert isinstance(spec_ranges_extract_attrs, (tuple, list)) and len(spec_ranges_extract_attrs) > 0 and all(
            map(lambda item: isinstance(item, SpecRangeExtractAttrs), spec_ranges_extract_attrs)), \
            "{} Error: Variable 'spec_ranges_extract_attrs' must be a tuple or list of 'SpecRangeExtractAttrs' " \
            "objects.".format(self.__class__.__name__)

        assert all([isinstance(item.name, str) and len(item.name) > 0 for item in spec_ranges_extract_attrs]), \
            "{} Error: Variable 'spec_ranges_extract_attrs' must contain elements whose 'name' property is a " \
            "non-empty string.".format(self.__class__.__name__)

        self.concat_axis = concat_axis
        self.spec_range_extractors = []
        for spec_range_attrs in spec_ranges_extract_attrs:
            ext = FingerSingleSpecRangePadisiH5ReaderDataExtractor(spec_range=spec_range_attrs.name,
                                                                   lambdas=spec_range_attrs.lambdas,
                                                                   crop_roi=crop_roi, roi=spec_range_attrs.roi,
                                                                   roi_relative_to=spec_range_attrs.roi_relative_to,
                                                                   auto_canon_target=auto_canon_target,
                                                                   target_slit_size=target_slit_size,
                                                                   lit_frames=spec_range_attrs.lit_frames,
                                                                   dark_frames=spec_range_attrs.dark_frames,
                                                                   mode=mode, bit_depth_norm=bit_depth_norm)
            self.spec_range_extractors.append(ext)

        # Make sure extractors are always sorted in the order of
        # FINGER_SPEC_RANGES: ('lsci', 'swir', 'vis', 'nir', 'bi') so that the order of the provided extractors does
        # not matter.
        self.spec_range_extractors = sorted(self.spec_range_extractors,
                                            key=lambda item: FINGER_SPEC_RANGES.index(item.spec_range))

    def __str__(self):
        return get_object_str_identifier(self)

    def get_property_extractors_validation_dictionary(self) -> (None, Callable, dict):
        return None

    def _extract_padisi_h5_data(self, padisi_h5_reader: PadisiH5Reader) -> np.ndarray:

        all_datasets = [spec_range_ext.extract_data(padisi_h5_reader)
                        for spec_range_ext in self.spec_range_extractors]

        # remove None datasets, which correspond to non-existing data (specifically missing dark channels)
        all_datasets = [ds for ds in all_datasets if ds is not None]
        concat_datasets = None
        if len(all_datasets) > 0:
            # reshape all datasets to prepare for concatenation
            reshape_shape = (-1 if self.concat_axis == 0 else 1, -1 if self.concat_axis == 1 else 1,
                             all_datasets[0].shape[2], all_datasets[0].shape[3])
            all_datasets = [ds.reshape(reshape_shape) for ds in all_datasets]
            concat_datasets = np.concatenate(all_datasets, axis=self.concat_axis)

        return concat_datasets


class FingerPadisiPreprocessedDataExtractor(PadisiH5ReaderDataExtractor):
    NEEDED_DICT_KEYS = ('data', 'identifiers')
    EXTRACTOR_NAME_FIXED_ORDER = ('FM', 'FS', 'FL', 'BN')
    EXTRACTOR_CHANNEL_MAPPING = {'FM': tuple(range(14, 21)),
                                 'FS': tuple(range(10, 14)),
                                 'FL': tuple(range(10)),
                                 'BN': tuple(range(21, 24))}

    def __init__(self, data_identifiers: (tuple, list), preprocessed_data_dict: dict):
        super(FingerPadisiPreprocessedDataExtractor, self).__init__()
        self.preprocessed_data_dict = preprocessed_data_dict
        self.data_identifiers = data_identifiers
        self._data_channels = sorted([item for sublist in
                                      [FingerPadisiPreprocessedDataExtractor.EXTRACTOR_CHANNEL_MAPPING[key]
                                       for key in self.data_identifiers] for item in sublist])
        # Pre-calculate indices per identifier for speed
        self._identifier_idx = dict(zip(self.preprocessed_data_dict['identifiers'],
                                        range(len(self.preprocessed_data_dict['identifiers']))))

    def _extract_padisi_h5_data(self, padisi_h5_reader: PadisiH5Reader) -> np.ndarray:
        reader_identifier = padisi_h5_reader.get_reader_identifier()
        assert reader_identifier in self._identifier_idx, \
            "{} Error: Identifier '{}' does not exist in " \
            "dataset.".format(self.__class__.__name__ + "._extract_padisi_h5_data()", reader_identifier)

        data = self.preprocessed_data_dict['data'][self._identifier_idx[reader_identifier], self._data_channels, :, :]
        if data.ndim < 4:
            data = np.expand_dims(data, axis=0)

        return data

    def get_property_extractors_validation_dictionary(self) -> (None, Callable, dict):
        pass

    @property
    def data_identifiers(self) -> tuple:
        return self._data_identifiers

    @data_identifiers.setter
    def data_identifiers(self, data_identifiers: (tuple, list)) -> None:
        assert isinstance(data_identifiers, (tuple, list)) and len(data_identifiers) > 0 and all(
            list(map(lambda item: item in FingerPadisiPreprocessedDataExtractor.EXTRACTOR_CHANNEL_MAPPING,
                     data_identifiers))) and len(set(data_identifiers)) == len(data_identifiers), \
            "{} Error: Property 'data_identifiers' must be a non-empty tuple or list containing unique strings: " \
            "[{}].".format(self.__class__.__name__,
                           ', '.join(["'" + item + "'" for item in
                                      FingerPadisiPreprocessedDataExtractor.EXTRACTOR_CHANNEL_MAPPING]))
        if isinstance(data_identifiers, list):
            data_identifiers = tuple(data_identifiers)
        self._data_identifiers = data_identifiers

    @property
    def preprocessed_data_dict(self) -> dict:
        return self._preprocessed_data_dict

    @preprocessed_data_dict.setter
    def preprocessed_data_dict(self, preprocessed_data_dict: dict) -> None:
        assert isinstance(preprocessed_data_dict, dict) and len(preprocessed_data_dict) > 0, \
            "{} Error: Property 'preprocessed_data_dict' must be a non-empty " \
            "dictionary.".format(self.__class__.__name__)
        missing_keys = [item for item in FingerPadisiPreprocessedDataExtractor.NEEDED_DICT_KEYS
                        if item not in preprocessed_data_dict]
        assert len(missing_keys) == 0, \
            "{} Error: Dictionary 'preprocessed_data_dict' is missing keys: " \
            "[{}].".format(self.__class__.__name__, ', '.join(["'" + item + "'" for item in missing_keys]))

        assert isinstance(preprocessed_data_dict['data'], np.ndarray) and preprocessed_data_dict['data'].ndim == 4, \
            "{} Error: Entry 'preprocessed_data_dict['data']' must be a 4-dimensional numpy " \
            "array.".format(self.__class__.__name__)

        assert isinstance(preprocessed_data_dict['identifiers'], list) and len(
            preprocessed_data_dict['identifiers']) and all(
            list(map(lambda item: isinstance(item, str) and len(item) > 0, preprocessed_data_dict['identifiers']))), \
            "{} Error: Entry 'preprocessed_data_dict['identifiers']' must be a non-empty list of " \
            "non-empty strings.".format(self.__class__.__name__)

        assert len(preprocessed_data_dict['identifiers']) == preprocessed_data_dict['data'].shape[0], \
            "{} Error: The number of elements of entry 'preprocessed_data_dict['identifiers']' and the fist " \
            "dimension of the numpy array 'preprocessed_data_dict['data']' must match".format(self.__class__.__name__)

        self._preprocessed_data_dict = preprocessed_data_dict
