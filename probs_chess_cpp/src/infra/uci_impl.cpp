#include "infra/uci_impl.h"

using namespace std;


namespace probs {


UciImpl::UciImpl(ConfigParser& config) : uci_player(config) {
    cout << "id name Probs v1.0" << "\n";
    cout << "id author Sergey P" << "\n";
    cout << "uciok" << "\n";
    cout << flush;
}


void UciImpl::Run() {
    while (true) {
        string input_line;
        getline(cin, input_line);

        stringstream line_stream(input_line);
        string command;
        line_stream >> command;

        if (command == "isready") {
            uci_player.WaitForReadyState();
            cout << "readyok" << "\n";
            cout << flush;
        }
        else if (command == "ucinewgame") {
            uci_player.OnNewGame();
        }
        else if (command == "position") {
            OnPositionCommand(line_stream);
        }
        else if (command == "go") {
            OnGoCommand(line_stream);
        }
        else if (command == "stop") {
            uci_player.Stop();
        }
        else if (command == "quit") {
            uci_player.Stop();
            break;
        }
        else if (command == "debug") {
            string on_off;
            line_stream >> on_off;
            if (on_off == "on")
                uci_player.SetDebug(true);
            else
                uci_player.SetDebug(false);
        }
        else if (command == "register") {
            cerr << "TODO: " << command << endl;
        }
        else if (command == "setoption") {
            cerr << "TODO: " << command << endl;
        }
        else if (command == "ponderhit") {
            cerr << "TODO: " << command << endl;
        }
        else if (command.empty()) {
            continue;
        }
        else {
            cerr << "Warning: Ignoring unknown command: '" << command << "'" << endl;
        }
    }
}


void UciImpl::OnPositionCommand(stringstream& line_stream) {
    string pos_string;
    vector<string> moves;
    bool is_reading_moves = false;
    while (line_stream) {
        string token;
        line_stream >> token;
        if (token.size() == 0)
            break;
        if (is_reading_moves)
            moves.push_back(token);
        else if (token == "startpos")
            pos_string = lczero::ChessBoard::kStartposFen;
        else if (token == "moves")
            is_reading_moves = true;
        else
            cerr << "WARNING: 'position' command unexpected token: '" << token << "'" << endl;
    }
    uci_player.SetPosition(pos_string, moves);
}


void UciImpl::OnGoCommand(stringstream& line_stream) {
    vector<string> search_moves;
    bool is_reading_moves = false;
    optional<chrono::milliseconds> wtime = nullopt;
    optional<chrono::milliseconds> btime = nullopt;
    optional<chrono::milliseconds> winc = nullopt;
    optional<chrono::milliseconds> binc = nullopt;
    optional<int> moves_to_go = nullopt;
    optional<int> depth = nullopt;
    optional<uint64_t> nodes = nullopt;
    optional<int> mate = nullopt;
    optional<chrono::milliseconds> fixed_time = nullopt;
    bool infinite = false;

    while (line_stream) {
        string token;
        line_stream >> token;
        if (token.size() == 0)
            break;
        if (token == "searchmoves")
            is_reading_moves = true;
        else if (token == "ponder") {
            cerr << "TODO: go ponder not supported" << endl;
        }
        else if (token == "wtime") {
            int x;
            line_stream >> x;
            wtime = std::chrono::milliseconds(x);
            is_reading_moves = false;
        }
        else if (token == "btime") {
            int x;
            line_stream >> x;
            btime = std::chrono::milliseconds(x);
            is_reading_moves = false;
        }
        else if (token == "winc") {
            int x;
            line_stream >> x;
            winc = std::chrono::milliseconds(x);
            is_reading_moves = false;
        }
        else if (token == "binc") {
            int x;
            line_stream >> x;
            binc = std::chrono::milliseconds(x);
            is_reading_moves = false;
        }
        else if (token == "movestogo") {
            int x;
            line_stream >> x;
            moves_to_go = x;
            is_reading_moves = false;
        }
        else if (token == "depth") {
            int x;
            line_stream >> x;
            depth = x;
            is_reading_moves = false;
        }
        else if (token == "nodes") {
            int x;
            line_stream >> x;
            nodes = x;
            is_reading_moves = false;
        }
        else if (token == "mate") {
            int x;
            line_stream >> x;
            mate = x;
            is_reading_moves = false;
        }
        else if (token == "movetime") {
            int x;
            line_stream >> x;
            fixed_time = std::chrono::milliseconds(x);
            is_reading_moves = false;
        }
        else if (token == "infinite") {
            infinite = true;
            is_reading_moves = false;
        }
        else if (is_reading_moves) {
            search_moves.push_back(token);
        }
        else
            cerr << "WARNING: 'go' command unexpected token: '" << token << "'" << endl;
    }

    uci_player.StartSearch(
        wtime,
        btime,
        winc,
        binc,
        moves_to_go,
        depth,
        nodes,
        mate,
        fixed_time,
        infinite,
        search_moves
    );
}


}  // namespace probs
