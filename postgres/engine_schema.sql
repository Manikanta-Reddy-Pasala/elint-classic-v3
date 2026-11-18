-- Postgres schema the engine works with (ids BIGINT). Loaded by ./elint.sh for every new database.
-- Taken from the stack that passed the 608-file run; postgres/init/ is the older box schema (integer ids).
--
-- PostgreSQL database dump
--


-- Dumped from database version 16.16 (Debian 16.16-1.pgdg13+1)
-- Dumped by pg_dump version 16.16 (Debian 16.16-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: clusters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.clusters (
    id integer NOT NULL,
    name character varying(255) DEFAULT ''::character varying NOT NULL,
    pdw_count integer DEFAULT 0,
    toa_ns bigint,
    signal_id integer,
    frequency_hz bigint,
    min_frequency_hz bigint,
    max_frequency_hz bigint,
    std_frequency_hz double precision,
    power_dbm double precision,
    min_power_dbm double precision,
    max_power_dbm double precision,
    std_power_dbm double precision,
    pulse_width_us double precision,
    min_pulse_width_us double precision,
    max_pulse_width_us double precision,
    std_pulse_width_us double precision,
    pulse_repetition_interval_us double precision,
    min_pulse_repetition_interval_us double precision,
    max_pulse_repetition_interval_us double precision,
    std_pulse_repetition_interval_us double precision,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    snr_db double precision,
    min_snr_db double precision,
    max_snr_db double precision,
    std_snr_db double precision,
    min_bandwidth_hz double precision,
    max_bandwidth_hz double precision,
    bandwidth_hz double precision,
    std_bandwidth_hz double precision,
    doa_deg double precision,
    min_doa_deg double precision,
    max_doa_deg double precision,
    std_doa_deg double precision,
    is_ai boolean DEFAULT true
);


--
-- Name: clusters_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.clusters ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.clusters_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: elint_clusters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.elint_clusters (
    cluster_id character varying(64) NOT NULL,
    name character varying(255) NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: elint_emitters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.elint_emitters (
    emitter_id character varying(64) NOT NULL,
    cluster_id character varying(64) NOT NULL,
    location character varying(255),
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: elint_signals; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.elint_signals (
    signal_id character varying(64) NOT NULL,
    emitter_id character varying(64) NOT NULL,
    band character varying(32),
    modulation character varying(32),
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: emitters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.emitters (
    id bigint NOT NULL,
    name character varying(255) NOT NULL,
    signature_matched_id integer,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    medoid double precision[],
    source text DEFAULT ''::text NOT NULL,
    freq double precision DEFAULT 0 NOT NULL,
    pri double precision DEFAULT 0 NOT NULL,
    doa double precision DEFAULT 0 NOT NULL,
    power double precision DEFAULT 0 NOT NULL,
    modulation text DEFAULT ''::text NOT NULL
);


--
-- Name: emitters_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.emitters ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.emitters_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: signals; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.signals (
    id bigint NOT NULL,
    frequency_hz double precision,
    bandwidth_hz double precision,
    receiver_id integer DEFAULT 1,
    power_dbm double precision,
    snr_db double precision,
    doa_deg double precision,
    emitter_id bigint,
    is_ai boolean DEFAULT true,
    modulation_type_id integer,
    toa_nanoseconds bigint,
    duration_ms double precision,
    emission_type character varying(32) DEFAULT 'PULSE'::character varying,
    toa_seconds timestamp with time zone DEFAULT now(),
    name text,
    freq double precision,
    pri double precision,
    doa double precision,
    power double precision,
    modulation text,
    count bigint DEFAULT 1 NOT NULL
);


--
-- Name: signals_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.signals ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.signals_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- Name: clusters clusters_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.clusters
    ADD CONSTRAINT clusters_pkey PRIMARY KEY (id);


--
-- Name: elint_clusters elint_clusters_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.elint_clusters
    ADD CONSTRAINT elint_clusters_pkey PRIMARY KEY (cluster_id);


--
-- Name: elint_emitters elint_emitters_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.elint_emitters
    ADD CONSTRAINT elint_emitters_pkey PRIMARY KEY (emitter_id);


--
-- Name: elint_signals elint_signals_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.elint_signals
    ADD CONSTRAINT elint_signals_pkey PRIMARY KEY (signal_id);


--
-- Name: emitters emitters_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.emitters
    ADD CONSTRAINT emitters_pkey PRIMARY KEY (id);


--
-- Name: signals signals_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.signals
    ADD CONSTRAINT signals_pkey PRIMARY KEY (id);


--
-- Name: elint_emitters elint_emitters_cluster_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.elint_emitters
    ADD CONSTRAINT elint_emitters_cluster_id_fkey FOREIGN KEY (cluster_id) REFERENCES public.elint_clusters(cluster_id);


--
-- Name: elint_signals elint_signals_emitter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.elint_signals
    ADD CONSTRAINT elint_signals_emitter_id_fkey FOREIGN KEY (emitter_id) REFERENCES public.elint_emitters(emitter_id);


--
-- PostgreSQL database dump complete
--


