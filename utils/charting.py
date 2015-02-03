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

# values is an 2D array so we can show side-by-side bars
def bar_chart(file_name, all_data_sets, title, x_labels, y_label, data_set_names,
              colors, group_padding=1, item_padding=0, show_amount=False):
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
            rectangles.extend(axis.bar(offset, [data_set[count]], bar_width, color=colors[data_set_index%len(colors)]))
            offset += bar_width + item_padding

        x_ticks.extend([start_offset + (offset - start_offset) / 2])

        # add a spacer
        offset += group_padding

    axis.set_xlim(0, offset)
    axis.set_ylim(0, max(sum(all_data_sets, [])) * 1.05)

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

def label_rectangle_footers(axis, rectangles):
    for rect in rectangles:
        height = rect.get_height()
        axis.text(rect.get_x()+rect.get_width()/2., .5, '%d'%int(height),
                ha='center', va='bottom')
