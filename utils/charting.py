# The MIT License (MIT)
# Copyright (c) 2015 Thoughtly
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.
#
#
#
# Used for generating various plots
import matplotlib.pyplot as plot
import logging

# values is an 2D array so we can show side-by-side bars
def bar_chart(file_name, all_data_sets, title, x_labels, y_label, data_set_names,
              colors, group_padding=1, item_padding=0, show_amount=False, max_height=None, edge_color="black"):

    num_data_sets = len(all_data_sets)
    data_set_length = len(all_data_sets[0])
    bar_width = 1.0

    figure, axis = plot.subplots()

    x_ticks = []
    offset = group_padding
    rectangles = []

    for count in range(0, data_set_length):
        start_offset = offset;
        for data_set_index, data_set in enumerate(all_data_sets):
            rectangles.extend(axis.bar(offset, [data_set[count]], bar_width, edgecolor=edge_color,
                                       color=colors[data_set_index%len(colors)]))
            offset += bar_width + item_padding

        x_ticks.extend([start_offset + (offset - start_offset) / 2])

        # add a spacer
        offset += group_padding

    axis.set_xlim(0, offset)

    if max_height is None:
        max_height = max(sum(all_data_sets, [])) * 1.05
    axis.set_ylim(0, max_height)

    if y_label is not None:
        axis.set_ylabel(y_label)

    if title is not None:
        axis.set_title(title)

    axis.set_xticks(x_ticks)
    if x_labels is not None:
        axis.set_xticklabels(x_labels)
    else:
        axis.set_xticklabels([""] * num_data_sets)

    if data_set_names is not None:
        axis.legend(rectangles, data_set_names)

    if show_amount:
        label_rectangle_footers(axis, rectangles)

    plot.savefig(file_name)
    plot.close("all")

def label_rectangle_footers(axis, rectangles):
    for rect in rectangles:
        height = rect.get_height()
        axis.text(rect.get_x()+rect.get_width()/2., .5, '%d'%int(height),
                ha='center', va='bottom')


def plot_distribution(file_name, title, y_label, data, num_buckets=None, bucket_size=None, show_bucket_values=True, color="blue", normalize=False):

    data_min = int(min(data))
    data_max = int(max(data)) + 1
    data_range = data_max - data_min

    if bucket_size is not None:
        num_buckets = int(data_range/bucket_size)
    else:
        bucket_size = float(data_range) / num_buckets

    # the entire range may not be covered since we cast num_buckets to an int,
    # so if we need a non-integer sized bucket this adds it (though can't add less than 1 bucket)
    if data_range - num_buckets * bucket_size:
        num_buckets += 1

    buckets = [0] * num_buckets
    bucket_ranges = []
    for i in range(0, num_buckets):
        bucket_ranges.extend([data_min + i*bucket_size])

    num_events = 0
    for item in data:
        bucket = int((item - data_min) / bucket_size);
        buckets[bucket] += 1
        num_events += 1

    if normalize:
        for index, value in enumerate(buckets):
            buckets[index] = float(value) / num_events

    if num_buckets > 100:
        edge_color = "none"
    else:
        edge_color = "black"

    #only show every N buckets so they all fit
    if show_bucket_values:
        step_size = 1 + len(bucket_ranges) / 20
        for i in range(0, len(bucket_ranges)):
            if (i%step_size) != 0:
                bucket_ranges[i] = ""
            else:
                bucket_ranges[i] = "{:.1f}".format(bucket_ranges[i])
    else:
        bucket_ranges = None

    bar_chart(file_name, [buckets], title, bucket_ranges, y_label, None, [color], 0, 0, edge_color=edge_color)

