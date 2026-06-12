#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mslab/book.hpp"
#include "mslab/execution_sim.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mslab_bindings, m) {
    m.doc() = "MicrostructureLab C++ bindings";

    // ── PriceLevel ────────────────────────────────────────────────────────────
    py::class_<mslab::PriceLevel>(m, "PriceLevel")
        .def_readonly("price", &mslab::PriceLevel::price)
        .def_readonly("size",  &mslab::PriceLevel::size)
        .def("__repr__", [](const mslab::PriceLevel& pl) {
            return "PriceLevel(price=" + std::to_string(pl.price) +
                   ", size="  + std::to_string(pl.size) + ")";
        });

    // ── OrderBook ─────────────────────────────────────────────────────────────
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
        .def("get_bids",        &mslab::OrderBook::get_bids, py::arg("n"))
        .def("get_asks",        &mslab::OrderBook::get_asks, py::arg("n"))
        .def_property_readonly("last_seq",      &mslab::OrderBook::last_seq)
        .def_property_readonly("sequence_gaps", &mslab::OrderBook::sequence_gaps)
        .def_property_readonly("bid_levels",    &mslab::OrderBook::bid_levels)
        .def_property_readonly("ask_levels",    &mslab::OrderBook::ask_levels)
        .def_property_readonly("symbol",        &mslab::OrderBook::symbol)
        .def("__repr__", [](const mslab::OrderBook& book) {
            return "OrderBook(symbol=" + std::string(book.symbol()) +
                   ", bids=" + std::to_string(book.bid_levels()) +
                   ", asks=" + std::to_string(book.ask_levels()) + ")";
        });

    // ── Enums ─────────────────────────────────────────────────────────────────
    py::enum_<mslab::Side>(m, "Side")
        .value("BUY",  mslab::Side::BUY)
        .value("SELL", mslab::Side::SELL)
        .export_values();

    py::enum_<mslab::OrderType>(m, "OrderType")
        .value("LIMIT",     mslab::OrderType::LIMIT)
        .value("IOC",       mslab::OrderType::IOC)
        .value("POST_ONLY", mslab::OrderType::POST_ONLY)
        .export_values();

    // ── Fill ──────────────────────────────────────────────────────────────────
    py::class_<mslab::Fill>(m, "Fill")
        .def_readonly("order_id",   &mslab::Fill::order_id)
        .def_readonly("side",       &mslab::Fill::side)
        .def_readonly("fill_price", &mslab::Fill::fill_price)
        .def_readonly("fill_size",  &mslab::Fill::fill_size)
        .def_readonly("fee",        &mslab::Fill::fee)
        .def_readonly("fill_ts_ns", &mslab::Fill::fill_ts_ns)
        .def_readonly("is_maker",   &mslab::Fill::is_maker)
        .def("__repr__", [](const mslab::Fill& f) {
            return "Fill(price=" + std::to_string(f.fill_price) +
                   ", size=" + std::to_string(f.fill_size) +
                   ", fee=" + std::to_string(f.fee) + ")";
        });

    // ── PositionSnapshot ──────────────────────────────────────────────────────
    py::class_<mslab::PositionSnapshot>(m, "PositionSnapshot")
        .def_readonly("position",     &mslab::PositionSnapshot::position)
        .def_readonly("avg_entry",    &mslab::PositionSnapshot::avg_entry)
        .def_readonly("realized_pnl", &mslab::PositionSnapshot::realized_pnl)
        .def_readonly("fee_drag",     &mslab::PositionSnapshot::fee_drag)
        .def_readonly("ts_ns",        &mslab::PositionSnapshot::ts_ns)
        .def("__repr__", [](const mslab::PositionSnapshot& p) {
            return "Position(pos=" + std::to_string(p.position) +
                   ", pnl=" + std::to_string(p.realized_pnl) +
                   ", fees=" + std::to_string(p.fee_drag) + ")";
        });

    // ── LatencyModel ──────────────────────────────────────────────────────────
    py::class_<mslab::LatencyModel>(m, "LatencyModel")
        .def(py::init<double, double, uint64_t>(),
             py::arg("base_ms")   = 10.0,
             py::arg("jitter_ms") = 0.0,
             py::arg("seed")      = 42)
        .def("delay_ms",       &mslab::LatencyModel::delay_ms)
        .def("delay_ns",       &mslab::LatencyModel::delay_ns)
        .def_property_readonly("base_ms",   &mslab::LatencyModel::base_ms)
        .def_property_readonly("jitter_ms", &mslab::LatencyModel::jitter_ms);

    // ── FeeModel ──────────────────────────────────────────────────────────────
    py::class_<mslab::FeeModel>(m, "FeeModel")
        .def(py::init<>())
        .def_readwrite("maker_fee", &mslab::FeeModel::maker_fee)
        .def_readwrite("taker_fee", &mslab::FeeModel::taker_fee);

    // ── RiskLimits ────────────────────────────────────────────────────────────
    py::class_<mslab::RiskLimits>(m, "RiskLimits")
        .def(py::init<>())
        .def_readwrite("max_position", &mslab::RiskLimits::max_position)
        .def_readwrite("max_drawdown", &mslab::RiskLimits::max_drawdown)
        .def_readonly("killed",        &mslab::RiskLimits::killed);

    // ── ExecutionSim ──────────────────────────────────────────────────────────
    py::class_<mslab::ExecutionSim>(m, "ExecutionSim")
        .def(py::init<const std::string&,
                      mslab::LatencyModel,
                      mslab::FeeModel,
                      mslab::RiskLimits>(),
             py::arg("symbol"),
             py::arg("latency") = mslab::LatencyModel(),
             py::arg("fees")    = mslab::FeeModel(),
             py::arg("limits")  = mslab::RiskLimits())
        .def("submit_order",    &mslab::ExecutionSim::submit_order,
             py::arg("side"), py::arg("type"), py::arg("price"),
             py::arg("size"), py::arg("signal_ts_ns"))
        .def("on_book_update",  &mslab::ExecutionSim::on_book_update,
             py::arg("price"), py::arg("size"), py::arg("is_bid"),
             py::arg("ts_ns"), py::arg("seq_start"), py::arg("seq_end"))
        .def("cancel_order",    &mslab::ExecutionSim::cancel_order,
             py::arg("order_id"))
        .def("reset",           &mslab::ExecutionSim::reset)
        .def("fills",           &mslab::ExecutionSim::fills)
        .def("position",        &mslab::ExecutionSim::position)
        .def("snapshots",       &mslab::ExecutionSim::snapshots)
        .def("is_killed",       &mslab::ExecutionSim::is_killed)
        .def("last_ts",         &mslab::ExecutionSim::last_ts)
        .def("__repr__", [](const mslab::ExecutionSim& s) {
            return "ExecutionSim(fills=" +
                   std::to_string(s.fills().size()) +
                   ", pos=" +
                   std::to_string(s.position().position) + ")";
        });
}