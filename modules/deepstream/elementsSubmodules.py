from modules.components.load_paths import *
from init import loadLogger
logger = loadLogger()

from dotenv import load_dotenv
load_dotenv(dotenv_path)

import gi
import sys
from gi.repository import Gst
from ctypes import *
# import sys
from common.is_aarch_64 import is_aarch64

rtsp_reconnect_interval = int(os.getenv("rtsp_reconnect_interval"))
file_loop = bool(os.getenv("file_loop"))
latency = int(os.getenv("latency"))
num_extra_surfaces = int(os.getenv("num_extra_surfaces"))
udp_buffer_size = int(os.getenv("udp_buffer_size"))
drop_frame_interval = int(os.getenv("drop_frame_interval"))
select_rtp_protocol = int(os.getenv("select_rtp_protocol"))

def cb_newpad(decodebin, decoder_src_pad,data):
    try:
        logger.info("In cb_newpad\n")
        caps = decoder_src_pad.get_current_caps() or decoder_src_pad.query_caps()
        gststruct=caps.get_structure(0)
        gstname=gststruct.get_name()
        source_bin=data
        features=caps.get_features(0)
    except Exception as e:
        logger.error("An error occurred while creating decoder caps", exc_info=e)
        
    try:
        # Need to check if the pad created by the decodebin is for video and not audio.
        logger.info("gstname=%s",gstname)
        if(gstname.find("video")!=-1):
            # Link the decodebin pad only if decodebin has picked nvidia decoder plugin nvdec_*. We do this by checking if the pad caps contain NVMM memory features.
            logger.info("features=%s",features)
            if features.contains("memory:NVMM"):
                # Get the source bin ghost pad
                bin_ghost_pad=source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
            else:
                sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")
    except Exception as e:
        logger.error("An error occurred while creating decoder pads", exc_info=e)

def decodebin_child_added(child_proxy,Object,name,user_data):
    try:
        logger.info("Decodebin child added: %s\n", name)
        if(name.find("decodebin") != -1):
            Object.connect("child-added",decodebin_child_added,user_data)
        # if "source" in name:
        #     source_element = child_proxy.get_by_name("source")
        #     if source_element.find_property('drop-on-latency') != None:
        #         Object.set_property("drop-on-latency", True)
                
        # if(name.find("nvv4l2decoder") != -1):
        #     if (is_aarch64()):
        #         Object.set_property("enable-max-performance", True)
        #         Object.set_property("drop-frame-interval", 0)
        #         Object.set_property("num-extra-surfaces", 0)
    except Exception as e:
        logger.error("An error occurred while adding decodebin child", exc_info=e)

def connect_sourcebin_to_decodebin(uri_decode_bin, nbin):
    try:
        # Connect to the "pad-added" signal of the decodebin which generates a
        # callback once a new pad for raw data has beed created by the decodebin
        uri_decode_bin.connect("pad-added",cb_newpad,nbin)
        uri_decode_bin.connect("child-added",decodebin_child_added,nbin)
        Gst.Bin.add(nbin,uri_decode_bin)
        bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
        return nbin
    except Exception as e:
        logger.error("An error occurred while connecting source pad to decode bin pad", exc_info=e)

def create_source_bin(index,uri):
    try:
        logger.info("Creating source bin")
        # Create a source GstBin to abstract this bin's content from the rest of the pipeline
        bin_name="source-bin-%02d" %index
        logger.info(bin_name)
        nbin=Gst.Bin.new(bin_name)
        if not nbin:
            sys.stderr.write(" Unable to create source bin \n")
    except Exception as e:
        logger.error("An error occurred while creating source bin", exc_info=e)

    try:
        # Source element for reading from the uri.
        # We will use decodebin and let it figure out the container format of the
        # stream and the codec and plug the appropriate demux and decode plugins.
        # use nvurisrcbin to enable file-loop
        uri_decode_bin=Gst.ElementFactory.make("nvurisrcbin", f"uri-decode-bin-{index}")
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.set_property("rtsp-reconnect-interval", rtsp_reconnect_interval)
        uri_decode_bin.set_property("file-loop", file_loop)
        # uri_decode_bin.set_property("udp-buffer-size",1048576)
        uri_decode_bin.set_property("latency", latency)
        uri_decode_bin.set_property("num-extra-surfaces", num_extra_surfaces)
        uri_decode_bin.set_property("udp-buffer-size", udp_buffer_size)
        uri_decode_bin.set_property("drop-frame-interval", drop_frame_interval)
        uri_decode_bin.set_property("select-rtp-protocol", select_rtp_protocol)
        if not is_aarch64():
            uri_decode_bin.set_property("cudadec-memtype", 2)
    except Exception as e:
        logger.error("An error occurred while setting nvurisrcbin properties", exc_info=e)

    try:
        return connect_sourcebin_to_decodebin(uri_decode_bin, nbin)
    except Exception as e:
        logger.error("An error occurred while connecting source bin to decode bin", exc_info=e)