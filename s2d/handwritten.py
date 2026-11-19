"""Set C: ten descriptions written freely, as a person would type them, with the expected flowchart written by
hand. Not generated, not used to tune the parser: the honest test of the text path."""
from .graph import Edge, Graph, Node


def G(start, steps, edges, decisions=()):
    nodes = [Node("S", start, "start")] + [Node(k, v, "process") for k, v in steps] + \
            [Node(k, v, "decision") for k, v in decisions] + [Node("E", "End", "end")]
    return Graph(nodes, [Edge(*e) if len(e) == 3 else Edge(e[0], e[1]) for e in edges])


CASES = [
    ("When a customer order arrives, the sales team checks the order and then reserves the stock. The parts are "
     "picked in the warehouse. After packing the parcel, the team ships the parcel. The process ends there.",
     G("a customer order arrives", [("A", "Check the order"), ("B", "Reserve the stock"), ("C", "Pick the parts"),
                                    ("D", "Pack the parcel"), ("F", "Ship the parcel")],
       [("S", "A"), ("A", "B"), ("B", "C"), ("C", "D"), ("D", "F"), ("F", "E")])),
    ("The process starts when a part arrives at the inspection station. The inspector measures the dimensions. If "
     "the part is within tolerance, the result is recorded; otherwise, the part is reworked and the dimensions are "
     "measured again. The process then ends.",
     G("a part arrives at the inspection station", [("A", "Measure the dimensions"), ("B", "Record the result"),
                                                    ("C", "Rework the part")],
       [("S", "A"), ("A", "D1"), ("D1", "B", "yes"), ("D1", "C", "no"), ("C", "A"), ("B", "E")],
       [("D1", "the part is within tolerance?")])),
    ("It starts when the machine is switched on. First we load the program, then we calibrate the machine. Next "
     "the operator starts the cycle. When the cycle is over, the tool is cleaned. The process is complete.",
     G("the machine is switched on", [("A", "Load the program"), ("B", "Calibrate the machine"),
                                      ("C", "Start the cycle"), ("D", "Clean the tool")],
       [("S", "A"), ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])),
    ("The process begins when a maintenance request is created. The planner approves the request. If the spare "
     "part is in stock, the technician replaces the filter; otherwise, the planner orders the spare part. Finally, "
     "the technician tests the pump. The process then ends.",
     G("a maintenance request is created", [("A", "Approve the request"), ("B", "Replace the filter"),
                                            ("C", "Order the spare part"), ("D", "Test the pump")],
       [("S", "A"), ("A", "D1"), ("D1", "B", "yes"), ("D1", "C", "no"), ("B", "D"), ("C", "D"), ("D", "E")],
       [("D1", "the spare part is in stock?")])),
    ("When a new batch is released, operators assemble the gearbox and weld the frame. The housing gets painted. "
     "A label is printed for each unit. The process ends.",
     G("a new batch is released", [("A", "Assemble the gearbox"), ("B", "Weld the frame"),
                                   ("C", "Paint the housing"), ("D", "Print the label")],
       [("S", "A"), ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])),
    ("The process starts when an invoice is received. Accounting checks the invoice. If the amount matches the "
     "order, the invoice is approved; if not, the supplier is contacted and the invoice is checked again. Then the "
     "invoice is archived. The process then ends.",
     G("an invoice is received", [("A", "Check the invoice"), ("B", "Approve the invoice"),
                                  ("C", "Contact the supplier"), ("D", "Archive the invoice")],
       [("S", "A"), ("A", "D1"), ("D1", "B", "yes"), ("D1", "C", "no"), ("C", "A"), ("B", "D"), ("D", "E")],
       [("D1", "the amount matches the order?")])),
    ("Everything starts when a weld is finished. The welder cleans the weld, then an inspector inspects the weld. "
     "Should the weld be rejected, it is ground out and redone; otherwise the report is updated. End of the process.",
     G("a weld is finished", [("A", "Clean the weld"), ("B", "Inspect the weld"), ("C", "Grind out the weld"),
                              ("D", "Update the report")],
       [("S", "A"), ("A", "B"), ("B", "D1"), ("D1", "C", "yes"), ("D1", "D", "no"), ("C", "D"), ("D", "E")],
       [("D1", "the weld is rejected?")])),
    ("The process starts when the alarm sounds. The operator stops the line and informs the supervisor. If the "
     "cause is found, the fault is fixed; otherwise, maintenance is called. The line is restarted. The process ends.",
     G("the alarm sounds", [("A", "Stop the line"), ("B", "Inform the supervisor"), ("C", "Fix the fault"),
                            ("D", "Call maintenance"), ("F", "Restart the line")],
       [("S", "A"), ("A", "B"), ("B", "D1"), ("D1", "C", "yes"), ("D1", "D", "no"), ("C", "F"), ("D", "F"),
        ("F", "E")], [("D1", "the cause is found?")])),
    ("When a sample is delivered, the lab registers the sample. Afterwards the sample is analysed. If the result is "
     "valid, the certificate is printed; otherwise, the sample is analysed again. The process then ends.",
     G("a sample is delivered", [("A", "Register the sample"), ("B", "Analyse the sample"),
                                 ("C", "Print the certificate")],
       [("S", "A"), ("A", "B"), ("B", "D1"), ("D1", "C", "yes"), ("D1", "B", "no"), ("C", "E")],
       [("D1", "the result is valid?")])),
    ("The process starts when a truck arrives. The goods are unloaded and counted. If the count is correct, the "
     "stock level is updated; otherwise, the supplier is contacted. Finally, the delivery note is archived. The "
     "process then ends.",
     G("a truck arrives", [("A", "Unload the goods"), ("B", "Count the goods"), ("C", "Update the stock level"),
                           ("D", "Contact the supplier"), ("F", "Archive the delivery note")],
       [("S", "A"), ("A", "B"), ("B", "D1"), ("D1", "C", "yes"), ("D1", "D", "no"), ("C", "F"), ("D", "F"),
        ("F", "E")], [("D1", "the count is correct?")])),
]
