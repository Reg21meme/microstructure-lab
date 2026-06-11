#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mslab/book.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mslab_bindings, m) {
    m.doc() = "MicrostructureLab C++ bindings";

    // Expose PriceLevel struct to Python
    py::class_<mslab::PriceLevel>(m, "PriceLevel")
        .def_readonly("price", &mslab::PriceLevel::price)
        .def_readonly("size",  &mslab::PriceLevel::size)
        .def("__repr__", [](const mslab::PriceLevel& pl) {
            return "PriceLevel(price=" + std::to_string(pl.price) +
                   ", size="  + std::to_string(pl.size) + ")";
        });

    // Expose OrderBook class to Python
    py::class_<mslab::OrderBook>(m, "OrderBook")
        .def(py::init<std::string>(), py::arg("symbol"))
        .def("apply_snapshot",  &mslab::OrderBook::apply_snapshot,
             py::arg("price"), py::arg("size"), py::arg("is_bid"))
        .def("set_snapshot_seq", &mslab::OrderBook::set_snapshot_seq,
             py::arg("seq"))
        .def("apply_update",    &mslab::OrderBook::apply_update,
             py::arg("price"), py::arg("size"), py::arg("is_bid"),
             py::arg("seq_start"), py::arg("seq_end"))
        .def("clear",           &mslab::OrderBook::clear)
        .def("best_bid",        &mslab::OrderBook::best_bid)
        .def("best_ask",        &mslab::OrderBook::best_ask)
        .def("spread",          &mslab::OrderBook::spread)
        .def("mid_price",       &mslab::OrderBook::mid_price)
        .def_property_readonly("last_seq",      &mslab::OrderBook::last_seq)
        .def_property_readonly("sequence_gaps", &mslab::OrderBook::sequence_gaps)
        .def_property_readonly("bid_levels",    &mslab::OrderBook::bid_levels)
        .def_property_readonly("ask_levels",    &mslab::OrderBook::ask_levels)
        .def_property_readonly("symbol",        &mslab::OrderBook::symbol)
        .def("get_bids", &mslab::OrderBook::get_bids, py::arg("n"))
        .def("get_asks", &mslab::OrderBook::get_asks, py::arg("n"))
        .def("__repr__", [](const mslab::OrderBook& book) {
            return "OrderBook(symbol=" + std::string(book.symbol()) +
                   ", bids=" + std::to_string(book.bid_levels()) +
                   ", asks=" + std::to_string(book.ask_levels()) + ")";
        });
}