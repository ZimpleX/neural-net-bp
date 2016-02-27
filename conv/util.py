def ff_next_img_size(prev_img, kern, padding, stride):
    """
    get the image size for next layer in the feed-forward case
    ARGUMENT:
        prev_img    tuple of (height, width)
        kern        integer of kernel size
    RETURN:
        tuple of next layer image (height, width)
    """
    prev_y, prev_x = prev_img
    assert (prev_y + 2*padding - kern)%stride == 0
    assert (prev_x + 2*padding - kern)%stride == 0
    cur_y = (prev_y + 2*padding - kern)/stride + 1
    cur_x = (prev_y + 2*padding - kern)/stride + 1
    return cur_y, cur_x


